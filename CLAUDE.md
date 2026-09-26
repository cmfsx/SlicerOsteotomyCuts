# OsteotomyCuts — 3D Slicer scripted extension

## Purpose
Virtual osteotomy tool for orthognathic and craniofacial surgical planning.
Slicer's existing tools only do single flat plane cuts. This module produces
multi-segment cuts as in Materialise Enlight / ProPlan CMF: the user places
points on the bone surface, a cutting sheet is extruded from them, and the
bone is split into separate fragments. Target cuts: BSSO (medial horizontal +
sagittal + vertical buccal, connected), Le Fort I, genioplasty, segmental cuts.

## Roadmap
- Phase 1: polyline cut (points on surface → extruded sheet → fragments)
- Phase 2: kerf thickness (saw blade width) and limited cut depth
- Phase 3: templates for BSSO / Le Fort I / genioplasty driven by landmarks
- Phase 4: interactive editing handles, fragment naming, STL export
Build one phase at a time. Design each so later phases need no rewrite.

## Environment
- Windows 11. Project: C:\Dev\OsteotomyCuts
- 3D Slicer 5.13.0 (preview build, 2026-06-30, r34833), installed for all users
- Slicer executable:
  C:\ProgramData\slicer.org\3D Slicer 5.13.0-2026-06-30\Slicer.exe
- Slicer Python interpreter:
  C:\ProgramData\slicer.org\3D Slicer 5.13.0-2026-06-30\bin\PythonSlicer.exe
- Python runs ONLY inside Slicer. Never pip install into system Python.
- New dependencies: ask me first. If approved, use slicer.util.pip_install()
  inside the module, guarded with a try/except import. Slicer is in
  ProgramData, so package installs may need admin rights.
- This is a preview build: prefer stable, long-standing Slicer APIs. If an API
  may differ between versions, say so and use the most compatible form.

## Architecture rules
- Standard Slicer split: Module / Widget / Logic / Test classes.
- All computation lives in Logic and must run headlessly without the GUI.
- Module state in a parameterNodeWrapper. No state stored on the widget.
- Everything persistent is a MRML node, so scenes save and reload correctly.
- UI changes go in Resources/UI/OsteotomyCuts.ui, not built in Python code.
- Remove all observers in cleanup(). No VTK object leaks.
- Never block the GUI thread; use progress dialogs for long operations.

## Cutting method (important)
- Core cut uses an implicit signed distance to the cutting sheet
  (vtkImplicitPolyDataDistance or equivalent) followed by clipping.
- Do NOT use mesh boolean libraries (vtkbool etc.) for the core cut —
  segmented CT bone meshes are rarely clean and booleans fail.
- Kerf is implemented as a distance threshold, not a boolean subtraction.
- Fragments are separated with vtkPolyDataConnectivityFilter.
- The original bone model is hidden, never deleted or modified.

## Testing
- Tests use synthetic geometry (vtkCubeSource, vtkSphereSource) or SampleData.
- NEVER read, open or copy any patient data folder or DICOM directory.
- Headless test run (PowerShell):
  & "C:\ProgramData\slicer.org\3D Slicer 5.13.0-2026-06-30\Slicer.exe" --no-splash --no-main-window --python-code "import slicer; slicer.util.selectModule('OsteotomyCuts'); import OsteotomyCuts; OsteotomyCuts.OsteotomyCutsTest().runTest(); slicer.util.exit(0)"

## Style
- British English in UI text and comments.
- PEP 8, type hints, docstrings on all Logic methods.
- Small, focused commits with clear messages.
- Before large changes, propose a plan and wait for approval.