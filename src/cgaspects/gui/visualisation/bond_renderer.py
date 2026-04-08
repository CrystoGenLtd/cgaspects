"""Instanced cylinder renderer for molecular bonds with per-half-bond colours.

Each bond A→B is split into two half-bond cylinder instances so the first half
is coloured with atom A's colour and the second half with atom B's colour.

Instance buffer layout (10 floats = 40 bytes per half-bond cylinder):
    start   [3] float32  – world-space start of this cylinder segment
    end     [3] float32  – world-space end of this cylinder segment
    color   [3] float32  – RGB in [0, 1]
    radius  [1] float32  – cylinder radius in Angstroms
"""

import numpy as np
from OpenGL.GL import GL_FLOAT, GL_TRIANGLES
from PySide6.QtGui import QOpenGLExtraFunctions
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLShader,
                              QOpenGLShaderProgram, QOpenGLVertexArrayObject)


class BondRenderer(QOpenGLExtraFunctions):
    """Instanced cylinder renderer for bond half-segments."""

    _INSTANCE_FLOATS = 10  # start(3) + end(3) + color(3) + radius(1)
    _INSTANCE_STRIDE = 40  # bytes

    # The vertex shader aligns a unit cylinder (centred at origin, z ∈ [-0.5, 0.5], r=1)
    # to the segment [start, end] using Rodrigues' rotation formula.
    vertex_shader_source = """
    #version 330 core

    // Base cylinder vertex (unit cylinder: r=1, z in [-0.5, 0.5])
    layout(location = 0) in vec3 vertexPosition;
    layout(location = 1) in vec3 vertexNormal;   // outward normal on unit cylinder

    // Per-instance attributes
    layout(location = 2) in vec3 iStart;
    layout(location = 3) in vec3 iEnd;
    layout(location = 4) in vec3 iColor;
    layout(location = 5) in float iRadius;

    out vec4 v_color;
    out vec3 v_normal;
    out vec3 v_position;

    uniform mat4 u_viewMat;
    uniform mat4 u_modelViewProjectionMat;
    uniform float u_pointSize;

    // Rotation matrix that maps vec3(0,0,1) onto the normalised direction d.
    mat3 alignZTo(vec3 d) {
        vec3 z = vec3(0.0, 0.0, 1.0);
        float cosA = dot(z, d);
        if (cosA > 0.9999) return mat3(1.0);
        if (cosA < -0.9999) {
            // 180-degree rotation around X
            return mat3(1.0, 0.0, 0.0,
                        0.0,-1.0, 0.0,
                        0.0, 0.0,-1.0);
        }
        vec3 axis = normalize(cross(z, d));
        float s = sqrt(1.0 - cosA * cosA);
        float t = 1.0 - cosA;
        return mat3(
            t*axis.x*axis.x + cosA,        t*axis.x*axis.y + s*axis.z,  t*axis.x*axis.z - s*axis.y,
            t*axis.x*axis.y - s*axis.z,    t*axis.y*axis.y + cosA,      t*axis.y*axis.z + s*axis.x,
            t*axis.x*axis.z + s*axis.y,    t*axis.y*axis.z - s*axis.x,  t*axis.z*axis.z + cosA
        );
    }

    void main() {
        vec3  diff  = iEnd - iStart;
        float len   = length(diff);
        if (len < 0.0001) { gl_Position = vec4(0.0); return; }
        vec3  dir   = diff / len;
        mat3  rot   = alignZTo(dir);

        // Scale: XY by iRadius (scaled by point_size), Z by len
        // iRadius is in Ångströms; u_pointSize/6.0 gives 1× at the default size of 6.
        float scaledRadius = iRadius * (u_pointSize / 6.0);
        vec3 localPos = vec3(vertexPosition.xy * scaledRadius,
                             (vertexPosition.z + 0.5) * len);
        vec3 worldPos = iStart + rot * localPos;

        // Normals: lateral face normals point radially outward in local XY
        //          cap normals are +/-Z in local space
        vec3 worldNorm = rot * vertexNormal;

        v_color    = vec4(iColor, 1.0);
        v_normal   = normalize(mat3(u_viewMat) * worldNorm);
        v_position = worldPos;
        gl_Position = u_modelViewProjectionMat * vec4(worldPos, 1.0);
    }
    """

    fragment_shader_source = """
    #version 330 core

    in vec4 v_color;
    in vec3 v_normal;
    in vec3 v_position;

    out vec4 fragColor;

    const vec3 lightDir   = normalize(vec3(0.2, 0.5, 1.0));
    const vec3 lightColor = vec3(1.0, 1.0, 1.0);

    void main() {
        vec3 norm = normalize(v_normal);
        float diff = min(0.3 + 0.7 * max(dot(norm, lightDir), 0.0), 1.0);
        fragColor = vec4(diff * lightColor * v_color.rgb, v_color.a);
    }
    """

    def __init__(self, gl):
        super().__init__()
        self.initializeOpenGLFunctions()
        self.instances = None
        self._n_base_verts = 0

        self.program = QOpenGLShaderProgram()
        self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, self.vertex_shader_source)
        self.program.addShaderFromSourceCode(QOpenGLShader.Fragment, self.fragment_shader_source)
        self.program.link()

        self.vao = QOpenGLVertexArrayObject()
        self.vao.create()
        self.vao.bind()

        # Base cylinder mesh (positions + normals interleaved, 6 floats/vertex)
        self.vertex_buffer = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vertex_buffer.create()
        self.vertex_buffer.bind()
        self.vertex_buffer.setUsagePattern(QOpenGLBuffer.StaticDraw)
        self._load_base_mesh(gl)

        base_stride = 6 * 4  # 6 floats × 4 bytes
        self.program.enableAttributeArray(0)  # position
        self.program.setAttributeBuffer(0, GL_FLOAT, 0, 3, base_stride)
        self.program.enableAttributeArray(1)  # normal
        self.program.setAttributeBuffer(1, GL_FLOAT, 12, 3, base_stride)
        self.vertex_buffer.release()

        # Per-instance buffer
        self.instance_buffer = QOpenGLBuffer()
        self.instance_buffer.create()
        self.instance_buffer.bind()
        self.instance_buffer.setUsagePattern(QOpenGLBuffer.DynamicDraw)

        stride = self._INSTANCE_STRIDE

        # location 2: start (3 floats, offset 0)
        self.program.enableAttributeArray(2)
        self.program.setAttributeBuffer(2, GL_FLOAT, 0, 3, stride)
        gl.glVertexAttribDivisor(2, 1)

        # location 3: end (3 floats, offset 12)
        self.program.enableAttributeArray(3)
        self.program.setAttributeBuffer(3, GL_FLOAT, 12, 3, stride)
        gl.glVertexAttribDivisor(3, 1)

        # location 4: color (3 floats, offset 24)
        self.program.enableAttributeArray(4)
        self.program.setAttributeBuffer(4, GL_FLOAT, 24, 3, stride)
        gl.glVertexAttribDivisor(4, 1)

        # location 5: radius (1 float, offset 36)
        self.program.enableAttributeArray(5)
        self.program.setAttributeBuffer(5, GL_FLOAT, 36, 1, stride)
        gl.glVertexAttribDivisor(5, 1)

        self.instance_buffer.release()
        self.vao.release()
        self.program.release()

    def _load_base_mesh(self, gl):
        """Build a flat-shaded unit cylinder and upload to the vertex buffer."""
        import trimesh

        # sections=10 is enough detail for bond cylinders
        cyl = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=10)

        # Expand to flat-shaded triangles (per-face normal repeated per vertex)
        verts = np.array(cyl.vertices, dtype=np.float32)
        faces = np.array(cyl.faces, dtype=np.int32)
        face_normals = np.array(cyl.face_normals, dtype=np.float32)

        tri_verts = verts[faces].reshape(-1, 3)  # (F*3, 3)
        tri_norms = np.repeat(face_normals, 3, axis=0)  # (F*3, 3)

        # Interleave: [pos(3), norm(3)] per vertex
        interleaved = np.hstack([tri_verts, tri_norms]).astype(np.float32).flatten()
        self._n_base_verts = tri_verts.shape[0]

        self.vertex_buffer.allocate(interleaved.tobytes(), interleaved.nbytes)

    def numberOfInstances(self):
        if self.instances is None:
            return 0
        return self.instances.size // self._INSTANCE_FLOATS

    def setBonds(self, instances: np.ndarray):
        """Upload bond half-cylinder instance data.

        instances : array-like, shape (N, 10)
            Columns: [start_x, start_y, start_z, end_x, end_y, end_z, r, g, b, radius]
        """
        if instances is None or len(instances) == 0:
            self.instances = None
            return
        self.instances = np.array(instances, dtype=np.float32).flatten()
        self.instance_buffer.bind()
        self.instance_buffer.allocate(self.instances.tobytes(), self.instances.nbytes)
        self.instance_buffer.release()

    def setUniforms(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, float):
                self.program.setUniformValue1f(k, v)
            elif isinstance(v, int):
                self.program.setUniformValue1i(k, v)
            else:
                self.program.setUniformValue(k, v)

    def bind(self, gl):
        self.program.bind()
        self.vao.bind()

    def release(self):
        self.vao.release()
        self.program.release()

    def draw(self, gl):
        if self.numberOfInstances() <= 0:
            return
        self.glDrawArraysInstanced(GL_TRIANGLES, 0, self._n_base_verts, self.numberOfInstances())
