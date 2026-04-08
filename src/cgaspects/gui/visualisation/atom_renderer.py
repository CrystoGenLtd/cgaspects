"""Instanced sphere renderer where each atom has its own VdW radius and CPK colour."""
import numpy as np
from OpenGL.GL import GL_FLOAT, GL_TRIANGLES
from PySide6.QtGui import QOpenGLExtraFunctions
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLShader,
                              QOpenGLShaderProgram, QOpenGLVertexArrayObject)


class AtomRenderer(QOpenGLExtraFunctions):
    """Renders a collection of atoms as instanced icospheres.

    Instance buffer layout (8 floats = 32 bytes per atom):
        position  [3] float32   – Cartesian world-space position
        color     [3] float32   – RGB in [0, 1]
        selected  [1] float32   – 0.0 or 1.0
        radius    [1] float32   – VdW radius in Angstroms (world-space)
    """

    _INSTANCE_FLOATS = 8  # floats per instance
    _INSTANCE_STRIDE = 32  # bytes per instance

    vertex_shader_source = """
    #version 330 core
    layout(location = 0) in vec3 vertexPosition;
    layout(location = 1) in vec3 position;
    layout(location = 2) in vec3 color;
    layout(location = 3) in float selected;
    layout(location = 4) in float atomRadius;

    out vec4 v_color;
    out vec3 v_normal;
    out vec3 v_position;
    out float v_selected;

    uniform mat4 u_viewMat;
    uniform mat4 u_modelViewProjectionMat;
    uniform float u_pointSize;

    void main() {
        v_selected = selected;
        // u_pointSize / 6.0 gives a 1× multiplier at the default point_size of 6.
        // atomRadius is in Ångströms (VdW radius × per-element scale override).
        float scale = (u_pointSize / 6.0) * atomRadius * (1.0 + selected * 0.15);

        mat4 transform = mat4(
            vec4(scale, 0.0, 0.0, 0.0),
            vec4(0.0, scale, 0.0, 0.0),
            vec4(0.0, 0.0, scale, 0.0),
            vec4(position, 1.0));
        mat4 normalTransform = inverse(transpose(transform));
        vec4 posTransformed = transform * vec4(vertexPosition, 1.0);

        v_normal   = normalize(mat3(u_viewMat) * mat3(normalTransform) * vertexPosition);
        v_position = posTransformed.xyz;
        v_color    = vec4(color, 1.0);
        gl_Position = u_modelViewProjectionMat * vec4(v_position, 1.0);
    }
    """

    fragment_shader_source = """
    #version 330 core

    in vec4  v_color;
    in vec3  v_normal;
    in vec3  v_position;
    in float v_selected;

    out vec4 fragColor;

    const vec3 lightDir   = normalize(vec3(0.2, 0.5, 1.0));
    const vec3 lightColor = vec3(1.0, 1.0, 1.0);
    const vec3 glowColor  = vec3(0.0, 0.9, 1.0);

    void main() {
        vec3 norm = normalize(v_normal);
        float diff = min(0.3 + 0.7 * max(dot(norm, lightDir), 0.0), 1.0);
        vec3 color = diff * lightColor * v_color.rgb;

        if (v_selected > 0.5) {
            vec3 viewDir = normalize(-v_position);
            float rim = pow(1.0 - max(dot(viewDir, norm), 0.0), 2.0);
            color = mix(color, glowColor, (0.6 + rim * 0.8) * 0.5);
            color += glowColor * rim * 0.6;
        }

        fragColor = vec4(color, v_color.a);
    }
    """

    def __init__(self, gl):
        super().__init__()
        self.initializeOpenGLFunctions()
        self.instances = None
        self.mesh = None
        self.vertices_flattened = None

        self.program = QOpenGLShaderProgram()
        self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, self.vertex_shader_source)
        self.program.addShaderFromSourceCode(QOpenGLShader.Fragment, self.fragment_shader_source)
        self.program.link()

        self.vao = QOpenGLVertexArrayObject()
        self.vao.create()
        self.vao.bind()

        # Base icosphere mesh (shared across all instances)
        self.vertex_buffer = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vertex_buffer.create()
        self.vertex_buffer.bind()
        self.vertex_buffer.setUsagePattern(QOpenGLBuffer.StaticDraw)
        self._load_base_mesh(gl)
        self.program.enableAttributeArray(0)
        self.program.setAttributeBuffer(0, GL_FLOAT, 0, 3, 12)
        self.vertex_buffer.release()

        # Per-instance buffer
        self.instance_buffer = QOpenGLBuffer()
        self.instance_buffer.create()
        self.instance_buffer.bind()
        self.instance_buffer.setUsagePattern(QOpenGLBuffer.DynamicDraw)

        stride = self._INSTANCE_STRIDE

        # location 1: position (3 floats, offset 0)
        self.program.enableAttributeArray(1)
        self.program.setAttributeBuffer(1, GL_FLOAT, 0, 3, stride)
        gl.glVertexAttribDivisor(1, 1)

        # location 2: color (3 floats, offset 12)
        self.program.enableAttributeArray(2)
        self.program.setAttributeBuffer(2, GL_FLOAT, 12, 3, stride)
        gl.glVertexAttribDivisor(2, 1)

        # location 3: selected (1 float, offset 24)
        self.program.enableAttributeArray(3)
        self.program.setAttributeBuffer(3, GL_FLOAT, 24, 1, stride)
        gl.glVertexAttribDivisor(3, 1)

        # location 4: atomRadius (1 float, offset 28)
        self.program.enableAttributeArray(4)
        self.program.setAttributeBuffer(4, GL_FLOAT, 28, 1, stride)
        gl.glVertexAttribDivisor(4, 1)

        self.instance_buffer.release()
        self.vao.release()
        self.program.release()

    def _load_base_mesh(self, gl):
        from trimesh.creation import icosphere
        self.mesh = icosphere(subdivisions=1)
        vertices = np.array(self.mesh.vertices, dtype=np.float32)
        faces = np.array(self.mesh.faces, dtype=np.uint32).flatten()
        self.vertices_flattened = vertices[faces, :].flatten()
        self.vertex_buffer.allocate(
            self.vertices_flattened.tobytes(), self.vertices_flattened.nbytes
        )

    def numberOfVertices(self):
        if self.vertices_flattened is None:
            return 0
        return self.vertices_flattened.size // 3

    def numberOfInstances(self):
        if self.instances is None:
            return 0
        return self.instances.size // self._INSTANCE_FLOATS

    def setPoints(self, points):
        """Upload atom instance data.

        points : array-like, shape (N, 8)
            Columns: [x, y, z, r, g, b, selected, radius]
        """
        self.instances = np.array(points, dtype=np.float32).flatten()
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
        self.glDrawArraysInstanced(GL_TRIANGLES, 0, self.numberOfVertices(), self.numberOfInstances())
