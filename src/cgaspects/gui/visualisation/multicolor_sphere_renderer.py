"""Impostor spheres split into coloured angular sections.

Used to visualise several atoms (or whole molecules) that occupy the same
lattice position.  Instead of drawing N overlapping single-colour spheres
(which z-fight into an indistinct blob), a single sphere is drawn and its
surface is partitioned into ``sectionCount`` longitudinal wedges, each shaded
with one overlapping atom's colour.  Two overlapping atoms therefore read as
two hemispheres, three as three wedges, and so on.

The section boundaries are computed from the *object-local* surface normal so
the coloured wedges are painted onto the sphere and rotate with the model,
rather than staying fixed to the screen.

Instance buffer layout (18 floats per sphere):
    position     [3] – Cartesian world-space position
    atomRadius   [1] – sphere radius in Angstroms (world-space)
    selected     [1] – 0.0 or 1.0
    sectionCount [1] – number of coloured wedges (1..MAX_SECTIONS)
    color0..3    [3]×4 – section colours (only the first sectionCount are used)
"""

from .impostor_spheres import ImpostorSphereRenderer
from .shading import LIGHTING_GLSL

MAX_SECTIONS = 4

_VERTEX_TEMPLATE = """
#version 330 core
layout(location = 0) in vec2 quadPos;   // billboard corner in [-1, 1]
//__INSTANCE_ATTRS__

flat out vec3  v_color0;
flat out vec3  v_color1;
flat out vec3  v_color2;
flat out vec3  v_color3;
flat out float v_count;
out float v_selected;
out float v_occlusion;
out vec3  v_centerView;
out vec3  v_fragView;
out float v_radiusView;

uniform mat4  u_modelViewMat;
uniform mat4  u_projectionMat;
uniform float u_pointSize;
uniform float u_scale;
uniform int   u_perspective;

void main() {
    float radiusModel = (__RADIUS_EXPR__) * (1.0 + selected * 0.15);
    vec3 centerView = (u_modelViewMat * vec4(position, 1.0)).xyz;
    float r = radiusModel * u_scale;

    float pad = 1.0;
    if (u_perspective == 1) {
        float d2 = dot(centerView, centerView);
        pad = (d2 > r * r) ? sqrt(d2 / (d2 - r * r)) : 2.0;
    }
    vec3 cornerView = centerView + vec3(quadPos * r * pad, 0.0);

    v_color0     = color0;
    v_color1     = color1;
    v_color2     = color2;
    v_color3     = color3;
    v_count      = sectionCount;
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

flat in vec3  v_color0;
flat in vec3  v_color1;
flat in vec3  v_color2;
flat in vec3  v_color3;
flat in float v_count;
in float v_selected;
in float v_occlusion;
in vec3  v_centerView;
in vec3  v_fragView;
in float v_radiusView;

out vec4 fragColor;

uniform mat4 u_projectionMat;
uniform mat4 u_modelViewMat;
uniform int  u_perspective;

//__LIGHTING__

const vec3  glowColor = vec3(0.0, 0.9, 1.0);
const float PI = 3.14159265359;

vec3 sectionColor(int idx) {
    if (idx <= 0) return v_color0;
    if (idx == 1) return v_color1;
    if (idx == 2) return v_color2;
    return v_color3;
}

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

    vec4 clip = u_projectionMat * vec4(hit, 1.0);
    gl_FragDepth = clamp((clip.z / clip.w) * 0.5 + 0.5, 0.0, 1.0);

    // Object-local normal: the coloured wedges are painted onto the sphere so
    // they rotate with the model instead of staying fixed to the screen.
    vec3 localN = normalize(inverse(mat3(u_modelViewMat)) * normal);
    int count = clamp(int(v_count + 0.5), 1, __MAX_SECTIONS__);
    float azimuth = atan(localN.z, localN.x);        // [-PI, PI]
    float frac = (azimuth + PI) / (2.0 * PI);         // [0, 1)
    int idx = clamp(int(floor(frac * float(count))), 0, count - 1);
    vec3 baseColor = sectionColor(idx);

    vec3 viewDir = (u_perspective == 1) ? -rd : vec3(0.0, 0.0, 1.0);
    vec3 color = shadeSurface(baseColor, normal, viewDir, v_occlusion);

    if (v_selected > 0.5) {
        float rim = pow(1.0 - max(dot(viewDir, normal), 0.0), 2.0);
        color = mix(color, glowColor, (0.6 + rim * 0.8) * 0.5);
        color += glowColor * rim * 0.6;
    }

    fragColor = vec4(color, 1.0);
}
"""


class MultiColorSphereRenderer(ImpostorSphereRenderer):
    """Impostor sphere whose surface is split into coloured angular wedges."""

    INSTANCE_ATTRS = (
        ("position", 3),
        ("atomRadius", 1),
        ("selected", 1),
        ("sectionCount", 1),
        ("color0", 3),
        ("color1", 3),
        ("color2", 3),
        ("color3", 3),
    )
    RADIUS_EXPR = "(u_pointSize / 6.0) * atomRadius"

    def _vertex_source(self) -> str:
        return _VERTEX_TEMPLATE.replace(
            "//__INSTANCE_ATTRS__", self._instance_attr_decls()
        ).replace("__RADIUS_EXPR__", self.RADIUS_EXPR)

    def _fragment_source(self) -> str:
        return _FRAGMENT_TEMPLATE.replace(
            "//__LIGHTING__", LIGHTING_GLSL
        ).replace("__MAX_SECTIONS__", str(MAX_SECTIONS))
