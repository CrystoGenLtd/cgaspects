import numpy as np
from OpenGL.GL import GL_FLOAT, GL_LINES
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLVertexArrayObject,
)


class AxesRenderer:
    point_size = 200.0

    def __init__(self):
        self.rendering_style = "cylinder"
        self.label_style = "x, y, z"
        self.length_multiplier = 1.0
        self.origin = (0.0, 0.0, 0.0)

        self.vertex_shader_source = """
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

        self.geometry_shader_line = """
        #version 330 core
        layout(lines) in;
        layout(triangle_strip, max_vertices = 4) out;

        uniform mat4 u_viewMat;
        uniform mat4 u_modelRotMat;
        uniform vec2 u_screenSize;
        uniform float u_axesThickness;

        in GS_IN {
            vec3 color;
        } gs_in[];

        out vec4 f_color;

        void main() {
            // Apply view matrix rotation to positions
            vec3 rotatedStart = mat3(u_viewMat) * mat3(u_modelRotMat) * gl_in[0].gl_Position.xyz * 0.1;
            vec3 rotatedEnd = mat3(u_viewMat) * mat3(u_modelRotMat) * gl_in[1].gl_Position.xyz * 0.1;

            // Correct for aspect ratio so axes don't stretch with window
            float aspect = u_screenSize.x / u_screenSize.y;
            rotatedStart.x /= aspect;
            rotatedEnd.x /= aspect;

            // Offset to corner
            vec2 screenPos = vec2(0.8, 0.8);
            rotatedStart.xy -= screenPos;
            rotatedEnd.xy -= screenPos;

            vec4 clipStart = vec4(rotatedStart, 1.0);
            vec4 clipEnd = vec4(rotatedEnd, 1.0);

            vec2 ndcStart = clipStart.xy;
            vec2 ndcEnd = clipEnd.xy;

            vec2 dir = normalize(ndcEnd - ndcStart);
            vec2 perpDir = vec2(-dir.y, dir.x);

            // Calculate line width in NDC
            float ndcThickness = u_axesThickness / u_screenSize.y * 2.0;

            vec2 offset = perpDir * ndcThickness / 2.0;
            gl_Position = vec4((ndcStart + offset), clipStart.zw);
            f_color = vec4(gs_in[0].color, 1.0);
            EmitVertex();

            gl_Position = vec4((ndcStart - offset), clipStart.zw);
            f_color = vec4(gs_in[0].color, 1.0);
            EmitVertex();

            gl_Position = vec4((ndcEnd + offset), clipEnd.zw);
            f_color = vec4(gs_in[1].color, 1.0);
            EmitVertex();

            gl_Position = vec4((ndcEnd - offset), clipEnd.zw);
            f_color = vec4(gs_in[1].color, 1.0);
            EmitVertex();

            EndPrimitive();
        }
        """

        self.geometry_shader_cylinder = """
        #version 330 core
        layout(lines) in;
        layout(triangle_strip, max_vertices = 128) out;

        uniform mat4 u_viewMat;
        uniform mat4 u_modelRotMat;
        uniform mat4 u_projMat;
        uniform vec2 u_screenSize;
        uniform float u_axesThickness;

        in GS_IN {
            vec3 color;
        } gs_in[];

        out vec4 f_color;
        out vec3 v_normal;
        out vec3 v_fragPos;

        const int segments = 12;
        const float PI = 3.14159265359;

        // Aspect ratio for correction
        float getAspect() {
            return u_screenSize.x / u_screenSize.y;
        }

        void emitCylinder(vec3 start, vec3 end, float radius, vec3 color) {
            vec3 dir = normalize(end - start);
            float aspect = getAspect();

            // Find perpendicular vectors
            vec3 up = abs(dir.y) < 0.99 ? vec3(0, 1, 0) : vec3(1, 0, 0);
            vec3 right = normalize(cross(up, dir));
            vec3 forward = cross(dir, right);

            // Generate cylinder
            for (int i = 0; i <= segments; i++) {
                float angle = float(i) * 2.0 * PI / float(segments);
                float ca = cos(angle);
                float sa = sin(angle);

                vec3 normal = normalize(right * ca + forward * sa);
                vec3 offset = normal * radius;

                // Bottom vertex
                vec3 pos1 = start + offset;
                vec3 rotated1 = mat3(u_viewMat) * mat3(u_modelRotMat) * pos1 * 0.1;
                rotated1.x /= aspect;
                rotated1.xy -= vec2(0.8, 0.8);
                gl_Position = vec4(rotated1, 1.0);
                v_normal = mat3(u_viewMat) * normal;
                v_fragPos = rotated1;
                f_color = vec4(color, 1.0);
                EmitVertex();

                // Top vertex
                vec3 pos2 = end + offset;
                vec3 rotated2 = mat3(u_viewMat) * mat3(u_modelRotMat) * pos2 * 0.1;
                rotated2.x /= aspect;
                rotated2.xy -= vec2(0.8, 0.8);
                gl_Position = vec4(rotated2, 1.0);
                v_normal = mat3(u_viewMat) * normal;
                v_fragPos = rotated2;
                f_color = vec4(color, 1.0);
                EmitVertex();
            }
            EndPrimitive();
        }

        void emitCone(vec3 base, vec3 tip, float baseRadius, vec3 color) {
            vec3 dir = normalize(tip - base);
            float aspect = getAspect();

            // Find perpendicular vectors
            vec3 up = abs(dir.y) < 0.99 ? vec3(0, 1, 0) : vec3(1, 0, 0);
            vec3 right = normalize(cross(up, dir));
            vec3 forward = cross(dir, right);

            // Generate cone
            for (int i = 0; i <= segments; i++) {
                float angle = float(i) * 2.0 * PI / float(segments);
                float ca = cos(angle);
                float sa = sin(angle);

                vec3 normal = normalize(right * ca + forward * sa + dir);
                vec3 offset = (right * ca + forward * sa) * baseRadius;

                // Base vertex
                vec3 pos1 = base + offset;
                vec3 rotated1 = mat3(u_viewMat) * mat3(u_modelRotMat) * pos1 * 0.1;
                rotated1.x /= aspect;
                rotated1.xy -= vec2(0.8, 0.8);
                gl_Position = vec4(rotated1, 1.0);
                v_normal = mat3(u_viewMat) * normal;
                v_fragPos = rotated1;
                f_color = vec4(color, 1.0);
                EmitVertex();

                // Tip vertex
                vec3 rotated2 = mat3(u_viewMat) * mat3(u_modelRotMat) * tip * 0.1;
                rotated2.x /= aspect;
                rotated2.xy -= vec2(0.8, 0.8);
                gl_Position = vec4(rotated2, 1.0);
                v_normal = mat3(u_viewMat) * dir;
                v_fragPos = rotated2;
                f_color = vec4(color, 1.0);
                EmitVertex();
            }
            EndPrimitive();
        }

        void main() {
            vec3 start = gl_in[0].gl_Position.xyz;
            vec3 end = gl_in[1].gl_Position.xyz;
            vec3 color = gs_in[0].color;

            // Calculate radius based on thickness
            float radius = u_axesThickness * 0.02;

            // Fixed arrowhead length (doesn't scale with length multiplier)
            float totalLength = length(end - start);
            float fixedArrowLength = 0.25; // Fixed size
            float arrowLength = min(fixedArrowLength, totalLength * 0.4); // Cap at 40% of total
            vec3 dir = normalize(end - start);
            vec3 shaftEnd = end - dir * arrowLength;

            // Draw cylinder shaft
            emitCylinder(start, shaftEnd, radius, color);

            // Draw cone arrowhead (25% of length, wider base)
            emitCone(shaftEnd, end, radius * 2.5, color);
        }
        """

        self.geometry_shader_arrow = """
        #version 330 core
        layout(lines) in;
        layout(triangle_strip, max_vertices = 10) out;

        uniform mat4 u_viewMat;
        uniform mat4 u_modelRotMat;
        uniform vec2 u_screenSize;
        uniform float u_axesThickness;

        in GS_IN {
            vec3 color;
        } gs_in[];

        out vec4 f_color;

        void main() {
            vec3 rotatedStart = mat3(u_viewMat) * mat3(u_modelRotMat) * gl_in[0].gl_Position.xyz * 0.1;
            vec3 rotatedEnd = mat3(u_viewMat) * mat3(u_modelRotMat) * gl_in[1].gl_Position.xyz * 0.1;

            // Correct for aspect ratio so axes don't stretch with window
            float aspect = u_screenSize.x / u_screenSize.y;
            rotatedStart.x /= aspect;
            rotatedEnd.x /= aspect;

            vec2 screenPos = vec2(0.8, 0.8);
            rotatedStart.xy -= screenPos;
            rotatedEnd.xy -= screenPos;

            vec4 clipStart = vec4(rotatedStart, 1.0);
            vec4 clipEnd = vec4(rotatedEnd, 1.0);

            vec2 ndcStart = clipStart.xy;
            vec2 ndcEnd = clipEnd.xy;

            vec2 dir = normalize(ndcEnd - ndcStart);
            vec2 perpDir = vec2(-dir.y, dir.x);

            float ndcThickness = u_axesThickness / u_screenSize.y * 2.0;

            // Fixed arrowhead length (doesn't scale with length multiplier)
            float totalLength = length(ndcEnd - ndcStart);
            float fixedArrowLength = 0.025; // Fixed size in NDC
            float arrowLength = min(fixedArrowLength, totalLength * 0.4); // Cap at 40% of total
            vec2 shaftEnd = ndcEnd - dir * arrowLength;
            vec2 offset = perpDir * ndcThickness / 2.0;

            gl_Position = vec4((ndcStart + offset), clipStart.zw);
            f_color = vec4(gs_in[0].color, 1.0);
            EmitVertex();

            gl_Position = vec4((ndcStart - offset), clipStart.zw);
            f_color = vec4(gs_in[0].color, 1.0);
            EmitVertex();

            gl_Position = vec4((shaftEnd + offset), clipStart.zw);
            f_color = vec4(gs_in[1].color, 1.0);
            EmitVertex();

            gl_Position = vec4((shaftEnd - offset), clipStart.zw);
            f_color = vec4(gs_in[1].color, 1.0);
            EmitVertex();

            EndPrimitive();

            // Draw arrowhead
            vec2 arrowOffset = perpDir * ndcThickness * 1.5;

            gl_Position = vec4((shaftEnd + arrowOffset), clipStart.zw);
            f_color = vec4(gs_in[1].color, 1.0);
            EmitVertex();

            gl_Position = vec4((shaftEnd - arrowOffset), clipStart.zw);
            f_color = vec4(gs_in[1].color, 1.0);
            EmitVertex();

            gl_Position = vec4(ndcEnd, clipEnd.zw);
            f_color = vec4(gs_in[1].color, 1.0);
            EmitVertex();

            EndPrimitive();
        }
        """

        self.fragment_shader_source = """
        #version 330 core

        in vec4 f_color;
        out vec4 fragColor;

        void main() {
            fragColor = f_color;
        }
        """

        self.fragment_shader_3d = """
        #version 330 core

        in vec4 f_color;
        in vec3 v_normal;
        in vec3 v_fragPos;
        out vec4 fragColor;

        void main() {
            // Simple directional lighting
            vec3 lightDir = normalize(vec3(0.5, 0.5, 1.0));
            vec3 normal = normalize(v_normal);

            // Ambient light (higher for brighter appearance)
            float ambient = 0.8;

            // Diffuse light
            float diff = max(dot(normal, lightDir), 0.0);
            float diffuse = 0.5 * diff;

            // Combine lighting
            float lighting = ambient + diffuse;

            fragColor = vec4(f_color.rgb * lighting, f_color.a);
        }
        """

        self.points = None
        self.crystallography = None  # Store crystallography object for fractional coords
        self.axes_thickness = 3.5  # Default thickness
        self.show_labels = True  # Whether to show axis labels
        self.label_color_same_as_axes = True  # Whether labels use axes colors
        self.label_color = (0.0, 0.0, 0.0)  # Default black for labels

        # Create shader programs for each rendering mode
        self.programs = {}

        # Line mode program
        self.programs["line"] = QOpenGLShaderProgram()
        self.programs["line"].addShaderFromSourceCode(
            QOpenGLShader.Vertex, self.vertex_shader_source
        )
        self.programs["line"].addShaderFromSourceCode(
            QOpenGLShader.Geometry, self.geometry_shader_line
        )
        self.programs["line"].addShaderFromSourceCode(
            QOpenGLShader.Fragment, self.fragment_shader_source
        )
        self.programs["line"].link()

        # Cylinder mode program (3D arrows)
        self.programs["cylinder"] = QOpenGLShaderProgram()
        self.programs["cylinder"].addShaderFromSourceCode(
            QOpenGLShader.Vertex, self.vertex_shader_source
        )
        self.programs["cylinder"].addShaderFromSourceCode(
            QOpenGLShader.Geometry, self.geometry_shader_cylinder
        )
        self.programs["cylinder"].addShaderFromSourceCode(
            QOpenGLShader.Fragment, self.fragment_shader_3d
        )
        self.programs["cylinder"].link()

        # Arrow mode program
        self.programs["arrow"] = QOpenGLShaderProgram()
        self.programs["arrow"].addShaderFromSourceCode(
            QOpenGLShader.Vertex, self.vertex_shader_source
        )
        self.programs["arrow"].addShaderFromSourceCode(
            QOpenGLShader.Geometry, self.geometry_shader_arrow
        )
        self.programs["arrow"].addShaderFromSourceCode(
            QOpenGLShader.Fragment, self.fragment_shader_source
        )
        self.programs["arrow"].link()

        # Default to cylinder program
        self.program = self.programs["cylinder"]

        self.vertex_buffer = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vertex_buffer.create()

        # Initialize with Cartesian axes
        self._initialize_axes()

    def _initialize_axes(self, axes_type="cartesian"):
        """Initialize axes based on type (cartesian or fractional)."""
        self.vertex_buffer.bind()

        if axes_type == "cartesian" or self.crystallography is None:
            # Standard Cartesian axes
            base_a = np.array([1, 0, 0])
            base_b = np.array([0, 1, 0])
            base_c = np.array([0, 0, 1])
        else:
            # Fractional axes - convert from fractional to Cartesian
            # Fractional unit vectors
            frac_a = np.array([1, 0, 0])
            frac_b = np.array([0, 1, 0])
            frac_c = np.array([0, 0, 1])

            # Convert to Cartesian using the transformation matrix
            cart_a = self.crystallography.frac_to_cart(frac_a.reshape(1, -1))[0]
            cart_b = self.crystallography.frac_to_cart(frac_b.reshape(1, -1))[0]
            cart_c = self.crystallography.frac_to_cart(frac_c.reshape(1, -1))[0]

            # Normalize for display
            max_length = max(np.linalg.norm(cart_a), np.linalg.norm(cart_b), np.linalg.norm(cart_c))
            base_a = cart_a / max_length
            base_b = cart_b / max_length
            base_c = cart_c / max_length

        # Apply length multiplier
        base_a = base_a * self.length_multiplier
        base_b = base_b * self.length_multiplier
        base_c = base_c * self.length_multiplier

        # Apply origin offset
        origin = np.array(self.origin, dtype=np.float32)

        self.points = np.array(
            (
                (origin[0], origin[1], origin[2], 1.0, 0.0, 0.0),
                (
                    origin[0] + base_a[0],
                    origin[1] + base_a[1],
                    origin[2] + base_a[2],
                    1.0,
                    0.0,
                    0.0,
                ),
                (origin[0], origin[1], origin[2], 0.0, 1.0, 0.0),
                (
                    origin[0] + base_b[0],
                    origin[1] + base_b[1],
                    origin[2] + base_b[2],
                    0.0,
                    1.0,
                    0.0,
                ),
                (origin[0], origin[1], origin[2], 0.0, 0.0, 1.0),
                (
                    origin[0] + base_c[0],
                    origin[1] + base_c[1],
                    origin[2] + base_c[2],
                    0.0,
                    0.0,
                    1.0,
                ),
            ),
            dtype=np.float32,
        ).flatten()

        self.vertex_buffer.allocate(self.points.tobytes(), self.points.nbytes)

        if not hasattr(self, "vao") or not self.vao.isCreated():
            self.vao = QOpenGLVertexArrayObject()
            self.vao.create()

        self.vao.bind()

        self.program.bind()
        self.program.enableAttributeArray(0)
        self.program.setAttributeBuffer(0, GL_FLOAT, 0, 3, 6 * 4)
        self.program.enableAttributeArray(1)
        self.program.setAttributeBuffer(1, GL_FLOAT, 3 * 4, 3, 6 * 4)

        self.vao.release()
        self.vertex_buffer.release()
        self.program.release()

    def set_crystallography(self, crystallography):
        """Set the crystallography object and update axes to fractional coordinates."""
        self.crystallography = crystallography
        self._initialize_axes("fractional")

    def set_cartesian(self):
        """Reset axes to Cartesian coordinates."""
        self.crystallography = None
        self._initialize_axes("cartesian")

    def set_axes_thickness(self, thickness):
        """Set the thickness of the axes lines."""
        self.axes_thickness = thickness

    def update_settings(self, settings):
        """Update axes rendering from settings dictionary."""
        if "style" in settings:
            self.rendering_style = settings["style"]
            # Switch to the appropriate shader program
            if self.rendering_style in self.programs:
                self.program = self.programs[self.rendering_style]
        if "thickness" in settings:
            self.axes_thickness = settings["thickness"]
        if "label_style" in settings:
            self.label_style = settings["label_style"]
        if "show_labels" in settings:
            self.show_labels = settings["show_labels"]
        if "label_color_same_as_axes" in settings:
            self.label_color_same_as_axes = settings["label_color_same_as_axes"]
        if "label_color" in settings:
            self.label_color = settings["label_color"]
        if "length_multiplier" in settings:
            self.length_multiplier = settings["length_multiplier"]
        if "origin" in settings:
            self.origin = settings["origin"]

        # Reinitialize axes with new settings
        if self.crystallography is None:
            self._initialize_axes("cartesian")
        else:
            self._initialize_axes("fractional")

    def get_axis_endpoints(self):
        """Get the endpoints of the axes for label placement."""
        if self.points is None:
            return []

        # Extract axis endpoints (every other point starting from index 1)
        # Points are: origin, end_a, origin, end_b, origin, end_c
        endpoints = []

        # Determine labels based on whether we're using fractional or Cartesian
        if self.crystallography is not None:
            labels = ["a", "b", "c"]
        else:
            labels = ["x", "y", "z"]

        for i, label in enumerate(labels):
            # Each axis is 2 points (origin and endpoint), 6 floats each
            endpoint_idx = (i * 2 + 1) * 6
            if endpoint_idx + 5 < len(self.points):
                endpoints.append(
                    {
                        "position": (
                            self.points[endpoint_idx],
                            self.points[endpoint_idx + 1],
                            self.points[endpoint_idx + 2],
                        ),
                        "label": label,
                        "color": (
                            self.points[endpoint_idx + 3],
                            self.points[endpoint_idx + 4],
                            self.points[endpoint_idx + 5],
                        ),
                    }
                )

        return endpoints

    def setUniforms(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, float):
                self.program.setUniformValue1f(k, v)
            elif isinstance(v, int):
                self.program.setUniformValue1i(k, v)
            else:
                self.program.setUniformValue(k, v)
        # Always set axes thickness
        self.program.setUniformValue1f("u_axesThickness", self.axes_thickness)

    def bind(self):
        self.program.bind()
        self.vao.bind()

    def release(self):
        self.vao.release()
        self.program.release()

    def draw(self, gl):
        n = self.numberOfPoints()
        gl.glDrawArrays(GL_LINES, 0, n)

    def numberOfPoints(self):
        if self.points is None:
            return 0
        return self.points.size // 6
