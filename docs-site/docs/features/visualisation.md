# Visualisation

CGAspects renders crystal structures as interactive 3D point clouds using OpenGL 3.3. This page covers all options that control how the point cloud is displayed.

---

## Rendering Styles

| Style | Description | Best For |
|-------|-------------|----------|
| Points | Fast GL_POINTS rasterization | Very large datasets (millions of points) |
| Spheres | GPU-instanced icospheres per point | Publication figures, small–medium crystals |
| Convex Hull | Solid surface enclosing all points | Morphology overview |
| Mesh | External 3D mesh file (OBJ/STL/PLY/GLB) | Custom geometry |

Switch between styles in the **Visualisation Settings** area.

---

## Coloring Points

Points can be coloured by any numerical column in the XYZ data:

| Colour Mode | Source Column |
|------------|--------------|
| Type | Molecule/atom type identifier |
| Number | Atom number |
| Layer | Growth layer index |
| Site | Site number |
| Energy | Particle energy |
| Single Colour | Fixed colour (no data mapping) |

### Colourmaps

When a numerical column is selected, a colourmap maps values to colours. Available colourmaps:

- Viridis (default)
- Cividis
- Plasma
- Inferno
- Magma
- Cool–Warm
- Grayscale

The colourmap range is set automatically to the min/max of the selected column for the current frame.

---

## Colour Legend

**View → Colour Legend** opens a dialog (stays open while you work) showing what values correspond to each colour in the current viewport.

- **Table view** — used when there are ≤ 10 unique values; shows each value and its colour swatch
- **Gradient view** — used when there are > 10 unique values; shows a continuous colour bar with min/max labels
- A **Toggle view** button switches between the two representations

The legend updates automatically when the colour mode or data column changes.

---

## Point Size

Adjust point size interactively:
- **Increase**: `Ctrl+=` or **View → Increase Point Size**
- **Decrease**: `Ctrl+-` or **View → Decrease Point Size**

For the **Spheres** style, the point size directly controls the sphere radius.

---

## Material & Lighting

**View → Sphere && Lighting Settings** opens a live, non-modal dialog controlling how spheres and atoms are shaded. Changes preview instantly and apply everywhere the scene is drawn — including exported images, animation frames and movies.

### Material presets

Pick a starting point from the **Preset** dropdown; editing any material field switches to **Custom**.

| Preset | Look |
|--------|------|
| Matte | Flat, no highlight (the historical default) |
| Plastic | Soft white highlight |
| Glossy | Tight bright highlight |
| Metallic | Colour-tinted highlight, darker body |
| Toon | Banded cel shading |

### Material fields

- **Ambient / Diffuse** — flat fill light vs. direction-dependent shading.
- **Specular / Shininess** — strength and tightness of the highlight.
- **Specular Tint** — `0` = white highlight (plastic), `1` = highlight tinted by the sphere colour (metal).
- **Toon Bands** — number of cel-shading bands (`0` = smooth).

### Lighting

The light is a *headlight* that follows the camera; **Azimuth** and **Elevation** offset it from the view direction.

- **Brightness** — overall exposure gain (`1.0` = unchanged). Values above `1` brighten the scene past the base colour, which ambient/diffuse alone cannot do. Applies to the viewport **and** to all exported stills, frames and movies.

### Ambient Occlusion

Enable to darken particles buried inside the crystal based on local neighbour density, adding depth to dense structures. **Strength** controls the effect. It is computed when data loads and may take a few seconds on large point sets.

---

## Site Highlighting

You can highlight specific lattice sites using **View → Highlight Sites** (`Ctrl+Shift+S`). This lets you colour-code individual sites or ranges while showing the rest of the crystal in a background colour.

See [Site Highlighting](site-highlighting.md) for full details.

---

## Background Colour

The background colour of the viewport can be changed from the Visualisation Settings. Black and white are common choices for publication figures.

---

## Export

**File → Export graphics…** (`Ctrl+E`) opens a chooser with three export types: **2D Image (PNG)**, **Ray-Traced Image (POV-Ray / Tachyon)**, and **3D Mesh**.

### Render to Image
The **2D Image (PNG)** option saves the current OpenGL viewport as a PNG. Resolution multiplier options (1×, 2×, 4×) allow high-DPI export. This is the fast, exact-match capture of what you see on screen.

### Ray-Traced Image (POV-Ray / Tachyon)
For photoreal offline rendering — with true shadows, ambient occlusion, reflections and depth of field — the **Ray-Traced Image** option snapshots the current scene (spheres, bonds, camera, light, background and material) and hands it to an external ray tracer.

- **Renderer** — POV-Ray or Tachyon. The dialog reports whether the binary is found on your `PATH`.
- **Quality** — *Match GL settings* maps the live [Material & Lighting](#material-lighting) one-to-one; *Photoreal* enables a second settings dialog (ambient occlusion / radiosity, soft shadows, reflection, focal blur, anti-aliasing).
- **Resolution** — defaults to the current viewport size.

If the renderer is not installed you can still **Export scene file** (`.pov` / `.dat`) and render it elsewhere. The material maps closely to each backend (e.g. Specular Tint → POV-Ray `metallic` / Tachyon metal phong); note that Toon banding is not reproduced by the ray tracers.

!!! note "Installing a renderer"
    POV-Ray: `brew install povray` (macOS) or your package manager. Tachyon ships bundled with [VMD](https://www.ks.uiuc.edu/Research/vmd/), or can be built from source; put a `tachyon` executable on your `PATH`.

The same **Renderer** and **Quality** options are available in the **Render Animation** dialog, so movies and PNG sequences can be ray traced frame-by-frame (much slower than the OpenGL renderer, but publication quality).

### Export 3D Mesh
The crystal geometry can be exported as a 3D mesh file:
- **OBJ** — Wavefront OBJ with surface normals
- **STL** — Binary STL for 3D printing
- **PLY** — Stanford PLY format
- **GLB** — glTF binary for web/game engines

### Export Point Cloud
**File → Export XYZ** (`Ctrl+Shift+E`) saves the current point cloud (including any deletions) as an XYZ file.
