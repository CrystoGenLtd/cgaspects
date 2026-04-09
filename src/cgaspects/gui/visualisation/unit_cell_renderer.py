"""Renders the 12 edges of a crystallographic unit cell box in world space."""

import numpy as np
from OpenGL.GL import GL_FLOAT, GL_LINES
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLVertexArrayObject,
)


# Eight corners of the unit cell in fractional coordinates
_FRAC_CORNERS = np.array(
    [
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1 – along a
        [0, 1, 0],  # 2 – along b
        [0, 0, 1],  # 3 – along c
        [1, 1, 0],  # 4 – along a+b
        [1, 0, 1],  # 5 – along a+c
        [0, 1, 1],  # 6 – along b+c
        [1, 1, 1],  # 7 – along a+b+c
    ],
    dtype=np.float64,
)

# 12 edges defined as (start_corner_idx, end_corner_idx, axis_index)
# axis_index: 0=a (red), 1=b (green), 2=c (blue)
_EDGES = [
    # along a
    (0, 1, 0), (2, 4, 0), (3, 5, 0), (6, 7, 0),
    # along b
    (0, 2, 1), (1, 4, 1), (3, 6, 1), (5, 7, 1),
    # along c
    (0, 3, 2), (1, 5, 2), (2, 6, 2), (4, 7, 2),
]

# One colour per axis (a=red, b=green, c=blue) – slightly muted
_AXIS_COLORS = np.array(
    [
        [0.85, 0.20, 0.20],  # a – red
        [0.20, 0.70, 0.20],  # b – green
        [0.20, 0.40, 0.85],  # c – blue
    ],
    dtype=np.float32,
)


class UnitCellRenderer:
    """Draws the 12 edges of a crystallographic unit cell as coloured lines."""

    def __init__(self, gl):
        self.visible = False
        self._vertices = None  # flat float32 array, None when no cell loaded

        vertex_src = """
        #version 330 core
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 color;

        out GS_IN {
            vec3 color;
        } gs_in;

        void main() {
            gs_in.color = color;
            gl_Position = vec4(position, 1.0);
        }
        """

        geometry_src = """
        #version 330 core
        layout(lines) in;
        layout(triangle_strip, max_vertices = 4) out;

        uniform mat4 u_projectionMat;
        uniform mat4 u_modelViewMat;
        uniform vec2 u_screenSize;
        uniform float u_lineScale;

        in GS_IN {
            vec3 color;
        } gs_in[];

        out vec4 f_color;

        void main() {
            vec4 startPos = u_modelViewMat * vec4(gl_in[0].gl_Position.xyz, 1.0);
            vec4 endPos   = u_modelViewMat * vec4(gl_in[1].gl_Position.xyz, 1.0);

            vec4 clipStart = u_projectionMat * startPos;
            vec4 clipEnd   = u_projectionMat * endPos;

            vec2 ndcStart = clipStart.xy / clipStart.w;
            vec2 ndcEnd   = clipEnd.xy   / clipEnd.w;

            vec2 dir     = normalize(ndcEnd - ndcStart);
            vec2 perpDir = vec2(-dir.y, dir.x);

            float ndcThickness = u_lineScale / u_screenSize.y * 2.0;
            vec2 offset = perpDir * ndcThickness / 2.0;

            gl_Position = vec4((ndcStart + offset) * clipStart.w, clipStart.zw);
            f_color = vec4(gs_in[0].color, 1.0);
            EmitVertex();

            gl_Position = vec4((ndcStart - offset) * clipStart.w, clipStart.zw);
            f_color = vec4(gs_in[0].color, 1.0);
            EmitVertex();

            gl_Position = vec4((ndcEnd + offset) * clipEnd.w, clipEnd.zw);
            f_color = vec4(gs_in[1].color, 1.0);
            EmitVertex();

            gl_Position = vec4((ndcEnd - offset) * clipEnd.w, clipEnd.zw);
            f_color = vec4(gs_in[1].color, 1.0);
            EmitVertex();

            EndPrimitive();
        }
        """

        fragment_src = """
        #version 330 core
        in vec4 f_color;
        out vec4 fragColor;

        void main() {
            fragColor = f_color;
        }
        """

        self.program = QOpenGLShaderProgram()
        self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, vertex_src)
        self.program.addShaderFromSourceCode(QOpenGLShader.Geometry, geometry_src)
        self.program.addShaderFromSourceCode(QOpenGLShader.Fragment, fragment_src)
        self.program.link()

        self.vao = QOpenGLVertexArrayObject()
        self.vao.create()
        self.vao.bind()

        self.vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vbo.create()
        self.vbo.bind()
        self.vbo.setUsagePattern(QOpenGLBuffer.DynamicDraw)

        stride = 6 * 4  # 6 floats × 4 bytes
        self.program.enableAttributeArray(0)
        self.program.setAttributeBuffer(0, GL_FLOAT, 0,       3, stride)
        self.program.enableAttributeArray(1)
        self.program.setAttributeBuffer(1, GL_FLOAT, 3 * 4,   3, stride)

        self.vbo.release()
        self.vao.release()
        self.program.release()

    # ------------------------------------------------------------------

    def set_cell(self, crystallography):
        """Upload the 12 unit-cell edges derived from *crystallography*.

        Parameters
        ----------
        crystallography : Crystallography
            Must have a valid ``direct`` matrix (rows = lattice vectors in
            Cartesian coordinates, frac→cart transform).
        """
        # Convert the 8 fractional corners to Cartesian
        cart = crystallography.frac_to_cart(_FRAC_CORNERS).astype(np.float32)

        # Build the vertex buffer: 12 edges × 2 vertices × 6 floats
        vertices = []
        for i0, i1, ax in _EDGES:
            color = _AXIS_COLORS[ax]
            vertices.append(np.concatenate([cart[i0], color]))
            vertices.append(np.concatenate([cart[i1], color]))

        self._vertices = np.array(vertices, dtype=np.float32).flatten()

        self.vbo.bind()
        self.vao.bind()
        self.vbo.allocate(self._vertices.tobytes(), self._vertices.nbytes)
        self.vao.release()
        self.vbo.release()

    def numberOfVertices(self):
        if self._vertices is None:
            return 0
        return self._vertices.size // 6  # 6 floats per vertex

    # ------------------------------------------------------------------

    def setUniforms(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, float):
                self.program.setUniformValue1f(k, v)
            elif isinstance(v, int):
                self.program.setUniformValue1i(k, v)
            else:
                self.program.setUniformValue(k, v)
        # Use a slightly thicker line than the default mesh edges
        self.program.setUniformValue1f("u_lineScale", 2.5)

    def bind(self):
        self.program.bind()
        self.vao.bind()

    def release(self):
        self.vao.release()
        self.program.release()

    def draw(self, gl):
        gl.glDrawArrays(GL_LINES, 0, self.numberOfVertices())
