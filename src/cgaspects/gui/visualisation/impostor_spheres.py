"""GPU sphere impostors: camera-facing quads ray-cast into analytic spheres.

Instead of tessellating each sphere into triangles (LOD meshes), every sphere
is drawn as a single 4-vertex billboard; the fragment shader intersects the
view ray with the analytic sphere, writes the exact depth, and shades the hit
point. Silhouettes are therefore pixel-perfect at any zoom level — resolution
scales automatically like a map renderer, with no LOD switching or popping —
while per-sphere vertex cost drops from 240 (subdivision-1 icosphere) to 4.

This is the technique used by molecular viewers such as PyMOL, VMD and Mol*.
"""

import numpy as np
from OpenGL.GL import GL_FLOAT, GL_TRIANGLE_STRIP
from PySide6.QtGui import QOpenGLExtraFunctions
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLShader,
                              QOpenGLShaderProgram, QOpenGLVertexArrayObject)

from .shading import LIGHTING_GLSL, compute_occlusion

_VERTEX_TEMPLATE = """
#version 330 core
layout(location = 0) in vec2 quadPos;   // billboard corner in [-1, 1]
//__INSTANCE_ATTRS__

out vec3  v_color;
out float v_selected;
out float v_occlusion;
out vec3  v_centerView;
out vec3  v_fragView;
out float v_radiusView;

uniform mat4  u_modelViewMat;
uniform mat4  u_projectionMat;
uniform float u_pointSize;
uniform float u_scale;        // uniform model-matrix scale (camera zoom)
uniform int   u_perspective;

void main() {
    // Selected spheres grow slightly so the glow ring reads at a glance.
    float radiusModel = (__RADIUS_EXPR__) * (1.0 + selected * 0.15);
    vec3 centerView = (u_modelViewMat * vec4(position, 1.0)).xyz;
    float r = radiusModel * u_scale;

    // In perspective the silhouette projects wider than the cross-section at
    // centre depth; pad the quad so it always covers the whole sphere.
    float pad = 1.0;
    if (u_perspective == 1) {
        float d2 = dot(centerView, centerView);
        pad = (d2 > r * r) ? sqrt(d2 / (d2 - r * r)) : 2.0;
    }
    vec3 cornerView = centerView + vec3(quadPos * r * pad, 0.0);

    v_color      = color;
    v_selected   = selected;
    v_occlusion  = occlusion;
    v_centerView = centerView;
    v_fragView   = cornerView;
    v_radiusView = r;
    gl_Position = u_projectionMat * vec4(cornerView, 1.0);
}
"""

_FRAGMENT_TEMPLATE = """
#version 330 core

in vec3  v_color;
in float v_selected;
in float v_occlusion;
in vec3  v_centerView;
in vec3  v_fragView;
in float v_radiusView;

out vec4 fragColor;

uniform mat4 u_projectionMat;
uniform int  u_perspective;

//__LIGHTING__

const vec3 glowColor = vec3(0.0, 0.9, 1.0);

void main() {
    vec3 ro, rd;
    if (u_perspective == 1) {
        ro = vec3(0.0);
        rd = normalize(v_fragView);
    } else {
        ro = vec3(v_fragView.xy, 0.0);
        rd = vec3(0.0, 0.0, -1.0);
    }

    vec3 oc = ro - v_centerView;
    float b = dot(rd, oc);
    float c = dot(oc, oc) - v_radiusView * v_radiusView;
    float disc = b * b - c;
    if (disc < 0.0) discard;
    float t = -b - sqrt(disc);
    if (t < 0.0) discard;

    vec3 hit = ro + t * rd;
    vec3 normal = (hit - v_centerView) / v_radiusView;

    // Exact sphere depth so impostors interleave correctly with meshes/bonds
    vec4 clip = u_projectionMat * vec4(hit, 1.0);
    gl_FragDepth = clamp((clip.z / clip.w) * 0.5 + 0.5, 0.0, 1.0);

    vec3 viewDir = (u_perspective == 1) ? -rd : vec3(0.0, 0.0, 1.0);
    vec3 color = shadeSurface(v_color, normal, viewDir, v_occlusion);

    if (v_selected > 0.5) {
        float rim = pow(1.0 - max(dot(viewDir, normal), 0.0), 2.0);
        color = mix(color, glowColor, (0.6 + rim * 0.8) * 0.5);
        color += glowColor * rim * 0.6;
    }

    fragColor = vec4(color, 1.0);
}
"""

_GLSL_TYPES = {1: "float", 2: "vec2", 3: "vec3"}


class ImpostorSphereRenderer(QOpenGLExtraFunctions):
    """Base instanced impostor renderer; subclasses define the instance layout.

    ``INSTANCE_ATTRS`` lists the per-instance attributes callers supply to
    :meth:`setPoints` (in order). An ``occlusion`` float is appended internally
    when ambient occlusion is enabled, so callers never provide it.
    """

    INSTANCE_ATTRS: tuple = (("position", 3), ("color", 3), ("selected", 1))
    # Model-space sphere radius; may reference u_pointSize and instance attrs.
    RADIUS_EXPR = "u_pointSize * 0.2"

    def __init__(self, gl):
        super().__init__()
        self.initializeOpenGLFunctions()
        self.instances = None
        self._raw_points = None
        self.ao_enabled = False
        self._in_floats = sum(size for _, size in self.INSTANCE_ATTRS)
        self._gpu_floats = self._in_floats + 1  # + occlusion

        self.program = QOpenGLShaderProgram()
        self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, self._vertex_source())
        self.program.addShaderFromSourceCode(QOpenGLShader.Fragment, self._fragment_source())
        self.program.link()

        self.vao = QOpenGLVertexArrayObject()
        self.vao.create()
        self.vao.bind()

        # Base quad (triangle strip), location 0
        quad = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=np.float32)
        self.vertex_buffer = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vertex_buffer.create()
        self.vertex_buffer.bind()
        self.vertex_buffer.setUsagePattern(QOpenGLBuffer.StaticDraw)
        self.vertex_buffer.allocate(quad.tobytes(), quad.nbytes)
        self.program.enableAttributeArray(0)
        self.program.setAttributeBuffer(0, GL_FLOAT, 0, 2, 8)
        self.vertex_buffer.release()

        # Per-instance buffer: caller attributes + occlusion, one location each
        self.instance_buffer = QOpenGLBuffer()
        self.instance_buffer.create()
        self.instance_buffer.bind()
        self.instance_buffer.setUsagePattern(QOpenGLBuffer.DynamicDraw)
        stride = self._gpu_floats * 4
        offset, loc = 0, 1
        for _, size in self.INSTANCE_ATTRS + (("occlusion", 1),):
            self.program.enableAttributeArray(loc)
            self.program.setAttributeBuffer(loc, GL_FLOAT, offset, size, stride)
            gl.glVertexAttribDivisor(loc, 1)
            offset += size * 4
            loc += 1
        self.instance_buffer.release()

        self.vao.release()
        self.program.release()

    def _instance_attr_decls(self) -> str:
        lines = []
        loc = 1
        for name, size in self.INSTANCE_ATTRS + (("occlusion", 1),):
            lines.append(f"layout(location = {loc}) in {_GLSL_TYPES[size]} {name};")
            loc += 1
        return "\n".join(lines)

    def _vertex_source(self) -> str:
        return _VERTEX_TEMPLATE.replace(
            "//__INSTANCE_ATTRS__", self._instance_attr_decls()
        ).replace("__RADIUS_EXPR__", self.RADIUS_EXPR)

    def _fragment_source(self) -> str:
        return _FRAGMENT_TEMPLATE.replace("//__LIGHTING__", LIGHTING_GLSL)

    # ------------------------------------------------------------------

    def setPoints(self, points):
        """Upload instance data; shape (N, in_floats) or flat multiple thereof."""
        points = np.asarray(points, dtype=np.float32).reshape(-1, self._in_floats)
        self._raw_points = points
        if self.ao_enabled and len(points):
            occ = compute_occlusion(points[:, :3])
        else:
            occ = np.zeros(len(points), dtype=np.float32)
        self.instances = np.ascontiguousarray(
            np.concatenate([points, occ[:, None]], axis=1)
        ).ravel()
        self.instance_buffer.bind()
        self.instance_buffer.allocate(self.instances.tobytes(), self.instances.nbytes)
        self.instance_buffer.release()

    def set_ao_enabled(self, enabled: bool):
        """Toggle ambient occlusion, re-uploading the cached instances."""
        enabled = bool(enabled)
        if enabled == self.ao_enabled:
            return
        self.ao_enabled = enabled
        if self._raw_points is not None and len(self._raw_points):
            self.setPoints(self._raw_points)

    def numberOfInstances(self):
        if self.instances is None:
            return 0
        return self.instances.size // self._gpu_floats

    def setUniforms(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, float):
                self.program.setUniformValue1f(k, v)
            elif isinstance(v, int):
                self.program.setUniformValue1i(k, v)
            else:
                self.program.setUniformValue(k, v)

    def bind(self, gl=None):
        self.program.bind()
        self.vao.bind()

    def release(self):
        self.vao.release()
        self.program.release()

    def draw(self, gl):
        if self.numberOfInstances() <= 0:
            return
        self.glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, self.numberOfInstances())
