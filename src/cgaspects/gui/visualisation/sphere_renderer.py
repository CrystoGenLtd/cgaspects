import numpy as np
from OpenGL.GL import GL_FLOAT, GL_TRIANGLES, GL_UNSIGNED_INT
from PySide6.QtGui import QOpenGLExtraFunctions
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLShader,
                              QOpenGLShaderProgram, QOpenGLVertexArrayObject)


class SphereRenderer(QOpenGLExtraFunctions):
    faces = None
    vertices = None
    instances = None

    def __init__(self, gl):
        super().__init__()
        self.initializeOpenGLFunctions()
        self.vertex_shader_source = """
        #version 330 core
        layout(location = 0) in vec3 vertexPosition;
        layout(location = 1) in vec3 position;
        layout(location = 2) in vec3 color;
        layout(location = 3) in float selected;

        out vec4 v_color;
        out vec3 v_normal;
        out vec3 v_position;
        out vec3 v_spherePosition;
        out float v_selected;
        uniform float u_pointSize;
        uniform mat4 u_viewMat;
        uniform mat4 u_modelViewProjectionMat;

        void main() {
          v_spherePosition = vertexPosition;
          v_selected = selected;

          // Scale up selected points slightly for glow effect
          float scale = u_pointSize * 0.2 * (1.0 + selected * 0.15);

          // to enable ellipsoids later keep this here for now
          mat4 transform = mat4(
            vec4(scale, 0, 0, 0),
            vec4(0, scale, 0, 0),
            vec4(0, 0, scale, 0),
            vec4(position, 1.0));
          mat4 normalTransform = inverse(transpose(transform));
          vec4 posTransformed = transform * vec4(vertexPosition, 1);

          v_normal = normalize(mat3(u_viewMat) * mat3(normalTransform) * vertexPosition);
          v_position = posTransformed.xyz;
          v_color = vec4(color, 1.0f);
          gl_Position = u_modelViewProjectionMat * vec4(v_position, 1.0);
        }
        """

        self.fragment_shader_source = """
        #version 330 core

        in vec4 v_color;
        in vec3 v_normal;
        in vec3 v_position;
        in vec3 v_spherePosition;
        in float v_selected;

        out vec4 fragColor;

        // Hard-coded light properties
        const vec3 lightDir = normalize(vec3(0.2, 0.5, 1.0));
        const vec3 lightColor = vec3(1.0, 1.0, 1.0);
        const vec3 glowColor = vec3(0.0, 0.9, 1.0);  // Cyan glow

        void main() {
            vec3 norm = normalize(v_normal);

            // Lambertian reflectance
            float diff = 0.7 * max(dot(norm, lightDir), 0.0);
            diff = min(0.3f + diff, 1.0f);

            // Combine diffuse lighting with vertex color
            vec3 color = diff * lightColor * v_color.rgb;

            // Add glow effect for selected points
            if (v_selected > 0.5) {
                // Fresnel-like rim glow effect
                vec3 viewDir = normalize(-v_position);
                float rim = 1.0 - max(dot(viewDir, norm), 0.0);
                rim = pow(rim, 2.0);

                // Add pulsing glow (can be animated with uniform time)
                float glowIntensity = 0.6 + rim * 0.8;
                color = mix(color, glowColor, glowIntensity * 0.5);

                // Add bright rim highlight
                color += glowColor * rim * 0.6;
            }

            fragColor = vec4(color, v_color.a);
        }
        """
        self.points = None
        self.program = QOpenGLShaderProgram()
        self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, self.vertex_shader_source)
        self.program.addShaderFromSourceCode(QOpenGLShader.Fragment, self.fragment_shader_source)
        self.program.link()

        self.vao = QOpenGLVertexArrayObject()
        self.vao.create()
        self.vao.bind()

        self.vertex_buffer = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vertex_buffer.create()
        self.vertex_buffer.bind()
        self.vertex_buffer.setUsagePattern(QOpenGLBuffer.StaticDraw)

        self.loadBaseMesh(gl)

        self.program.enableAttributeArray(0)
        self.program.setAttributeBuffer(0, GL_FLOAT, 0, 3, 3 * 4)
        self.vertex_buffer.release()

        self.instance_buffer = QOpenGLBuffer()
        self.instance_buffer.create()
        self.instance_buffer.bind()
        self.instance_buffer.setUsagePattern(QOpenGLBuffer.DynamicDraw)

        # Stride: 3 floats (position) + 3 floats (color) + 1 float (selected) = 7 floats = 28 bytes
        stride = 28

        offset = 0
        self.program.enableAttributeArray(1)
        self.program.setAttributeBuffer(1, GL_FLOAT, offset, 3, stride)
        gl.glVertexAttribDivisor(1, 1)

        offset = 12  # 3 floats * 4 bytes
        self.program.enableAttributeArray(2)
        self.program.setAttributeBuffer(2, GL_FLOAT, offset, 3, stride)
        gl.glVertexAttribDivisor(2, 1)

        offset = 24  # 6 floats * 4 bytes
        self.program.enableAttributeArray(3)
        self.program.setAttributeBuffer(3, GL_FLOAT, offset, 1, stride)
        gl.glVertexAttribDivisor(3, 1)

        self.instance_buffer.release()
        self.vao.release()
        self.program.release()

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
        self.glDrawArraysInstanced(
            GL_TRIANGLES, 0, self.numberOfVertices(), self.numberOfInstances()
        )

    def numberOfInstances(self):
        if self.instances is None:
            return 0
        return self.instances.size // 7  # 3 (position) + 3 (color) + 1 (selected)

    def numberOfFaces(self):
        if self.mesh is None:
            return 0
        return self.mesh.faces.size // 3

    def numberOfVertices(self):
        if self.mesh is None:
            return 0
        return self.vertices_flattened.size // 3

    def setPoints(self, points):
        self.instances = np.array(points, dtype=np.float32).flatten()
        self._updateBuffers()

    def _updateBuffers(self):
        self.instance_buffer.bind()
        self.instance_buffer.allocate(self.instances.tobytes(), self.instances.nbytes)
        self.instance_buffer.release()

    def loadBaseMesh(self, gl, **kwargs):
        from trimesh.creation import icosphere

        self.mesh = icosphere(subdivisions=1)
        self.vertices = np.array(self.mesh.vertices, dtype=np.float32)
        self.faces = np.array(self.mesh.faces, dtype=np.uint32).flatten()
        self.vertices_flattened = self.vertices[self.faces, :].flatten()
        self.vertex_buffer.allocate(
            self.vertices_flattened.tobytes(), self.vertices_flattened.nbytes
        )
