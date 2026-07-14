# Unit Cell Viewer

The Unit Cell Viewer is a standalone 3D window for inspecting the **crystal net** — the unit cell, the molecules (tiles) inside it, and how each molecule connects to its neighbours through faces. Unlike the main viewport (which shows the simulated crystal shape as it grows), this viewer shows the underlying **periodic structure** the simulation is built from.

Open it from **Crystallography → Unit Cell Viewer**. It becomes available once a structure file with molecule templates has been loaded.

---

## Data Source

Everything the viewer draws — the unit cell, the molecule templates, and the connectivity between them — comes from the **CrystoGen structure file** that is auto-loaded with your simulation data (see [File Formats](../file-formats.md#crystogen-structure-file)). No separate net file is required.

The structure file's net section lists, for every tile (molecule) in the unit cell, its face-neighbours and the relative unit-cell offset of each neighbour, e.g. a tile might connect to "tile 3, one cell over along *a*, one cell back along *b*". The viewer reads this directly, so connections are correct even when several molecules share the same molecular formula.

### Net File (Energy Overlay)

A separate CrystoGen **net file** can be imported on top via **Import Net File…**. It does not add new connections — it *annotates* the ones already parsed from the structure file with interaction **distance (r)** and **energy**, which then appear on the connection labels in the selection tree and can drive [connection tube thickness](#connections-as-tubes).

The importer matches net-file molecules to structure tiles in two passes: first by matching molecule order one-to-one, then, if that fails, by matching each net molecule's label to the structure tile's formula. If the interaction counts don't line up, the import is rejected with a message rather than silently mismatching energies to the wrong bonds.

Use **Clear Net** to drop the energy overlay without losing the underlying connectivity.

---

## Supercell

The **Supercell** panel has one spinbox per axis (*a*, *b*, *c*, 1–6 cells each). Increasing any of them grows the scene to a repeating grid of unit cells, replicating every molecule and connection accordingly — useful for seeing how a local interaction pattern tiles through the lattice.

Each cell of the supercell is an independent, individually selectable/hideable instance of every molecule (see [Selection](#selection--visibility) below).

---

## Display

The **Display** group toggles what's drawn without changing any underlying selection:

| Toggle | Effect |
|--------|--------|
| Show Unit Cell | The cell edge box(es) |
| Show Molecules | Atoms and intramolecular bonds |
| Show Connections | The lines/tubes linking neighbouring molecules |
| Show Neighbours Outside Supercell | Also draw the "ghost" molecules that boundary connections point to, one cell beyond the current supercell |

**Show Connections From** lists every tile that has net connectivity; untick a tile to stop fanning connection lines out from it (its own molecule stays visible if Show Molecules is on).

---

## Selection & Visibility

The **Molecules / Connections** tree lists every tile (with one entry per supercell cell once more than one cell is grown), every atom, and every connection. Selecting an item highlights it in the viewport.

- **Hide** — hide just the selected items.
- **Isolate** — hide everything *except* the selection. Isolating a molecule keeps its connections visible; isolating a connection keeps both endpoint molecules visible — isolating never blanks the very thing you selected.
- **Show All** — clear all hiding.

Long connection labels (e.g. `M1 → M3 (-1,1,-1) r=3.89 Å E=-8.000` once a net file is imported) scroll horizontally rather than being clipped.

---

## Appearance & Export

**Appearance / Export…** opens a separate, non-modal dialog so it can stay open alongside the 3D view:

### Connections

Choose the connection drawing **Style**:

- **Lines** — thin coloured segments, one colour per interaction shell (grouped by energy when a net file is loaded, otherwise by centroid–centroid distance).
- **Tubes (scaled by energy)** — cylinders whose radius scales with the magnitude of the interaction energy, so the strongest interactions read as visibly thicker. Requires an imported net file to have real energies; without one, all tubes use the minimum radius.

### Sizes

- **Atom Radius Scale** — scales all atom VdW radii.
- **Bond Radius (Å)** — intramolecular bond cylinder radius.
- **Connection Radius Scale** — scales connection line width / tube radius (tube radii keep their relative energy scaling).

### Sphere & Lighting Settings

Opens the same [Material & Lighting](visualisation.md#material-lighting) dialog used by the main viewport — material presets, ambient/diffuse/specular/shininess, specular tint, toon banding, headlight azimuth/elevation/brightness, and ambient occlusion. Changes preview live and apply to exported images, ray traces and animation frames.

### Export

- **Export Image (PNG)…** — saves the current view at 1×, 2×, or 4× the viewport resolution, matching **File → Export graphics…** in the main window.
- **Export Ray-Traced Image (POV-Ray/Tachyon)…** — snapshots the scene (atoms, bonds, and any energy tubes) into a POV-Ray or Tachyon scene and renders it externally for true shadows, reflections and depth of field. See [Visualisation → Export](visualisation.md#ray-traced-image-pov-ray-tachyon) for renderer setup and quality options — the same dialog and options are used here.

---

## Animation

**Show Animation Timeline** reveals the same [keyframe timeline](animation.md) used by the main viewport, embedded below the 3D view. Add keyframes to capture camera position/orientation, preview the interpolated camera motion, and render to an MP4 or PNG sequence via OpenGL or a ray tracer — identical workflow to [Animation & Movie Rendering](animation.md), scoped to this viewer's camera. Since the unit cell viewer has no growth timeline, keyframes only animate the camera, not a data frame.

---

## Navigation

- **Left-drag** — rotate the view.
- **Scroll** — zoom in/out.
- **Reset View** — re-fit the camera to the current scene.
