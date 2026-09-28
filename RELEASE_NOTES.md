# Osteotomy Cuts — release notes

## 1.0.0

First public release: virtual osteotomies for orthognathic and craniofacial surgical planning in 3D Slicer.

> Research and planning software, not a medical device; use is entirely at the user's own risk. See [DISCLAIMER.md](DISCLAIMER.md).

### Features

- **Osteotomy lines** of any shape, placed on the bone surface, cut along a saw direction taken from the 3D view or from a direction line.
- **Saw blade thickness** (ideal cut, thin 0.5 mm, standard 1.0 mm or custom), **cut depth**, and **cut reach beyond the line** to stop a cut in a gap between bones.
- **Several osteotomy lines in one step**, each with its own saw direction and limits; later lines stop where they meet earlier ones (e.g. horizontal and pterygomaxillary cuts of a Le Fort I).
- **Closed (watertight) bone segments**, with a check of every segment and a warning when one is not closed; small loose pieces between cuts are joined to the neighbouring segment.
- **Solid bone models** (*Create solid bone model*, *Treat bone as solid*): holes sealed, marrow and canals filled, so that segmented bone divides cleanly.
- **Model check before cutting**: a bone model that is not closed, has internal surfaces or several pieces is reported, with the option to make it solid.
- **Live preview** of the cut with red lines wherever it comes out of the bone.
- **Structures to protect** (nerve canal, tooth roots, other): distance from the cut allowing for the blade, closest point marked, preview coloured green / amber / red, and a warning before cutting.
- **Provenance** recorded on every bone segment (lines, saw direction, blade, depth, solid settings, time, module version, git commit).
- **Undo** of a whole osteotomy and **rejoining** of bone segments; scenes save and reload with all settings.
- Surgeon-friendly labels, three-line tooltips (what / typical value / technical term) and a quick start in the module help.
- Terms of use shown on first use of each version.

### Requirements

- 3D Slicer 5.8 or later (developed and tested on 5.13.0 preview, 2026-06-30, Windows 11). No extra Python packages: VTK, NumPy and SciPy ship with Slicer.

### Known limitations

- Solid bone models are rebuilt from voxels: the surface moves by up to about half the detail (0.1 mm at 0.25 mm), and gaps narrower than about twice the gap sealing are filled.
- A cut ending inside the bone is refused (its slot ends cannot be closed yet); a later osteotomy line must meet an earlier line within that line's cut.
- Distances to structures are as accurate as their segmentation or tracing.
- Landmark-driven templates (BSSO, Le Fort I, genioplasty) are planned for a later version.

### Verification

- 86 headless tests on synthetic geometry, all passing (88 s).
- Benchmark of the cutting core, all capped cuts watertight (including a 1.1 million triangle synthetic jaw, 4.6 s).
- Checks on de-identified mandible, canal and teeth models: all bone segments closed; the canal correctly reported as entered by a body cut.
- Licence scan (ScanCode Toolkit 32.5.0): only GPL-3.0 licensing, no third-party code.

### Licence

GPL-3.0-or-later ([LICENSE](LICENSE)); a commercial licence is available on request. Contributions require the [Contributor Licence Agreement](CLA.md).
