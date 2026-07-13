# Animation & Movie Rendering

CGAspects can produce camera-fly-through and time-evolution movies of a crystal using a **keyframe timeline**. You capture the view at several points, and the app smoothly interpolates the camera (and, optionally, the growth frame) between them, then renders the result to a video or image sequence.

All animation actions live under the **Animation** menu.

---

## The Keyframe Timeline

Open the timeline with **Animation → Keyframe Timeline** (`Ctrl+T`). It docks below the 3D viewport and has three parts:

- **Toolbar** — `+ Add Keyframe`, `▶ Preview`, **Duration** (0.5–3600 s), **FPS** (1–120), and `Render…`.
- **Track** — the ruler with draggable keyframe diamonds and a scrubbable playhead. Each segment between keyframes is labelled with its interpolation mode. Right-click a keyframe to **Delete** or **Duplicate** it.
- **Inspector** — edits the selected keyframe's **Time**, **Data frame**, and **Interpolation to next**.

---

## Keyframes

A keyframe is a snapshot of the current view. **Animation → Add Keyframe Here** (`K`), or the `+ Add Keyframe` button, captures:

- Camera position and orientation
- Point size
- Any crystallographic planes and directions
- An optional **data frame** (see below)

Build an animation by posing the view, adding a keyframe, moving/advancing, adding another, and so on. You need at least **two** keyframes to render.

### Data frame (animating growth)

Each keyframe can pin a specific **growth frame** of a multi-frame XYZ dataset via the inspector's **Data frame** field. Set different frames on successive keyframes to animate the crystal *growing* while the camera moves. Leave it unset to hold the current frame.

### Interpolation

Each segment between two keyframes has its own easing, chosen in the inspector's **Interpolation to next**:

| Mode | Motion |
|------|--------|
| `linear` | Constant speed |
| `ease_in_out` | Slow start and end (default) |
| `ease_in` | Slow start |
| `ease_out` | Slow end |
| `constant` | Hold, then jump at the next keyframe |

---

## Preview

Click **▶ Preview** (or scrub the playhead) to play the animation live in the viewport at the current duration and FPS, without rendering. Use it to check timing and framing before committing to a render.

---

## Rendering to Video

Open **Animation → Render Animation…** (or the timeline's `Render…` button) to configure the output:

- **Output** — file path (MP4) or folder (PNG sequence).
- **Format** — **MP4 Video** or **PNG Sequence**. MP4 encoding (via `imageio-ffmpeg`) is bundled with CGAspects, so both options are available out of the box.
- **Resolution** — defaults to the current viewport size.
- **Frame rate / Frames** — shown from the timeline's FPS and duration.
- **Renderer** — see below.

Progress is shown as each frame is rendered; you can **Stop** partway through.

### Renderer: OpenGL vs Ray Traced

The **Renderer** dropdown chooses how each frame is drawn:

| Renderer | Speed | Look |
|----------|-------|------|
| OpenGL (fast) | Real-time | Exactly what the viewport shows |
| POV-Ray (ray traced) | Slow (per-frame) | Photoreal: true shadows, ambient occlusion, reflections |
| Tachyon (ray traced) | Slow (per-frame) | Photoreal alternative |

When a ray-traced renderer is selected, a **Quality** option appears — *Match GL settings* mirrors the live [Material & Lighting](visualisation.md#material-lighting), or *Photoreal* enables the extra settings dialog (ambient occlusion/radiosity, soft shadows, reflection, focal blur, anti-aliasing). Each frame is exported to a scene file and rendered by the external program, so ray-traced movies are **much slower** than OpenGL but produce publication-quality results. Ray-traced movies also honour the **Resolution** setting exactly.

!!! note "Installing a ray tracer"
    The chosen renderer must be on your `PATH`. POV-Ray: `brew install povray` (macOS) or your package manager. Tachyon ships bundled with [VMD](https://www.ks.uiuc.edu/Research/vmd/) or can be built from source. If neither is installed, use the OpenGL renderer or the still-image [ray-trace export](visualisation.md#ray-traced-image-pov-ray-tachyon). See [Visualisation → Export](visualisation.md#export) for details.

---

## Saving & Loading Animations

A timeline (keyframes, interpolation, duration, FPS) can be saved to a JSON file and reloaded later:

- **Animation → Save Animation…** — write the current timeline to `.json`.
- **Animation → Load Animation…** — restore a saved timeline.

This lets you re-render the same animation at different resolutions or with a different renderer without rebuilding the keyframes.

---

## Menu Reference

| Action | Shortcut | Description |
|--------|----------|-------------|
| Keyframe Timeline | `Ctrl+T` | Show or hide the timeline dock |
| Add Keyframe Here | `K` | Capture the current view as a keyframe |
| Render Animation… | — | Open the video/PNG-sequence render dialog |
| Save Animation… | — | Save the timeline to a JSON file |
| Load Animation… | — | Load a timeline from a JSON file |
