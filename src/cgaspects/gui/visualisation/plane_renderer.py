"""Renderer for crystallographic planes in world space.

Renders planes as semi-transparent quads using the MVP matrix.
"""

import numpy as np
from OpenGL.GL import GL_FLOAT, GL_TRIANGLES
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLVertexArrayObject,
)


class PlaneRenderer:
    def __init__(self):
        self.planes = []
        self.points = None

        self.vertex_shader_source = """
        #version 330 core
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;
        layout(location = 2) in vec3 color;
        layout(location = 3) in float alpha;

        uniform mat4 u_modelViewProjectionMat;
        uniform mat4 u_viewMat;
        uniform mat4 u_modelMat;

        out vec4 f_color;
        out vec3 v_normal;
        out vec3 v_fragPos;

        void main() {
            gl_Position = u_modelViewProjectionMat * vec4(position, 1.0);
            v_normal = mat3(u_viewMat) * mat3(u_modelMat) * normal;
            v_fragPos = position;
            f_color = vec4(color, alpha);
        }
        """

        self.fragment_shader_source = """
        #version 330 core
        in vec4 f_color;
        in vec3 v_normal;
        in vec3 v_fragPos;
        out vec4 fragColor;

        void main() {
            vec3 lightDir = normalize(vec3(0.5, 0.5, 1.0));
            vec3 normal = normalize(v_normal);

            // Two-sided lighting
            float diff = abs(dot(normal, lightDir));
            float ambient = 0.6;
            float diffuse = 0.4 * diff;
            float lighting = ambient + diffuse;

            fragColor = vec4(f_color.rgb * lighting, f_color.a);
        }
        """

        self.program = QOpenGLShaderProgram()
        self.program.addShaderFromSourceCode(
            QOpenGLShader.Vertex, self.vertex_shader_source
        )
        self.program.addShaderFromSourceCode(
            QOpenGLShader.Fragment, self.fragment_shader_source
        )
        self.program.link()

        self.vertex_buffer = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vertex_buffer.create()
        self.vertex_buffer.bind()

        self.vao = QOpenGLVertexArrayObject()
        self.vao.create()
        self.vao.bind()

        stride = 10 * 4  # 10 floats * 4 bytes
        self.program.bind()
        self.program.enableAttributeArray(0)
        self.program.setAttributeBuffer(0, GL_FLOAT, 0, 3, stride)
        self.program.enableAttributeArray(1)
        self.program.setAttributeBuffer(1, GL_FLOAT, 3 * 4, 3, stride)
        self.program.enableAttributeArray(2)
        self.program.setAttributeBuffer(2, GL_FLOAT, 6 * 4, 3, stride)
        self.program.enableAttributeArray(3)
        self.program.setAttributeBuffer(3, GL_FLOAT, 9 * 4, 1, stride)

        self.vao.release()
        self.vertex_buffer.release()
        self.program.release()

    def set_planes(self, planes, crystallography=None):
        """Set plane data from list of plane dicts.

        Each dict has: normal, origin, fractional, size, color, alpha
        """
        self.planes = planes
        if not planes:
            self.points = None
            return

        # Build vertex data: each plane is 2 triangles (6 vertices)
        # Vertex format: x, y, z, nx, ny, nz, r, g, b, alpha = 10 floats
        all_points = []

        for p in planes:
            normal = np.array(p.normal, dtype=np.float64)
            origin = np.array(p.origin, dtype=np.float64)
            size = p.size
            r, g, b = p.color
            alpha = p.alpha

            # Convert Miller indices (hkl) to Cartesian normal via reciprocal lattice.
            # frac_to_cart uses the direct lattice (correct for directions [uvw]);
            # planes need M^{-T} since (hkl) lives in reciprocal space.
            if p.fractional and crystallography is not None:
                normal = crystallography.miller_to_cart_normal(normal)

            # Normalize
            length = np.linalg.norm(normal)
            if length < 1e-10:
                continue
            normal = normal / length

            # Build two perpendicular vectors to create quad
            up = np.array([0, 1, 0]) if abs(normal[1]) < 0.99 else np.array([1, 0, 0])
            right = np.cross(up, normal)
            right = right / np.linalg.norm(right) * size
            forward = np.cross(normal, right)
            forward = forward / np.linalg.norm(forward) * size

            # Four corners of the plane quad
            v0 = origin - right - forward
            v1 = origin + right - forward
            v2 = origin + right + forward
            v3 = origin - right + forward

            # Two triangles: v0-v1-v2 and v0-v2-v3
            for v in [v0, v1, v2, v0, v2, v3]:
                all_points.extend([
                    v[0], v[1], v[2],
                    normal[0], normal[1], normal[2],
                    r, g, b, alpha,
                ])

        if not all_points:
            self.points = None
            return

        self.points = np.array(all_points, dtype=np.float32)
        self._upload_data()

    def _upload_data(self):
        if self.points is None or len(self.points) == 0:
            return

        self.vertex_buffer.bind()
        self.vertex_buffer.allocate(self.points.tobytes(), self.points.nbytes)
        self.vertex_buffer.release()

    def setUniforms(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, float):
                self.program.setUniformValue1f(k, v)
            elif isinstance(v, int):
                self.program.setUniformValue1i(k, v)
            else:
                self.program.setUniformValue(k, v)

    def bind(self):
        self.program.bind()
        self.vao.bind()

    def release(self):
        self.vao.release()
        self.program.release()

    def draw(self, gl):
        if self.points is None or len(self.points) == 0:
            return
        n = len(self.points) // 10
        gl.glDrawArrays(GL_TRIANGLES, 0, n)

    def numberOfVertices(self):
        if self.points is None:
            return 0
        return len(self.points) // 10
