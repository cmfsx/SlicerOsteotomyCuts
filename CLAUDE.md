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
- Phase 2: kerf thickness (saw blade width), limited cut depth, and capping
  of fragment cut faces (watertight fragments)
- Phase 3: templates for BSSO / Le Fort I / genioplasty driven by landmarks
- Phase 4: interactive editing handles, fragment naming, STL export
Build one phase at a time. Design each so later phases need no rewrite.

## Current status (updated 2026-09-28)
- Phase 1: complete.
- Phase 2: complete (steps 1-5) and confirmed by the user in Slicer on
  2026-09-27 (grooves, folded/closed grooves, through-cut across a groove,
  real bone).
- v1.0.0 pre-release (branch `release`, from 3a7458d = Phase 2, no Phase 3):
  Parts 0-10 of the user's to-do list, done in order, stopping after each
  for testing in Slicer. Phase 3 work continues on `master` and is merged
  after the release (Part 10). Nothing is pushed publicly, tagged or put
  into a pull request until the user types exactly: GO RELEASE.
  Done and committed on `release` (2026-09-28), each tested headlessly:
  Part 0 (Phase 2 robustness fixes from master), 0b (limited cuts, cut
  reach beyond the line, red outline), 1 (solid bone models), 0d (several
  osteotomy lines in one step; later lines stop at earlier ones; settings
  per line), joining of small loose pieces, 2 (model / segment closedness
  checks and warnings), 3 (structures to protect), 4 (provenance), 5
  (surgeon wording; the user's Help / Acknowledgement / contributors text),
  6 (GPL-3.0 LICENSE, SPDX headers, DISCLAIMER.md + first-use terms, CLA),
  7 (README, metadata, COMPATIBILITY.md), 8 (audit, RELEASE_NOTES.md).
  Confirmed by the user in Slicer up to Part 3. SAFETY_NOTE and the Help /
  Acknowledgement texts are the user's exact wording: do not reword them.
  DISCLAIMER.md and CLA.md are drafts awaiting the user's / a lawyer's
  review. NEXT: the user's Slicer check, then Part 9 only after GO RELEASE,
  then Part 10 (merge into master). The approved plan (Parts 0-10) is in
  the user's Claude plans folder: twinkly-brewing-phoenix.md.
- To resume in a new session: check `git branch --show-current` is
  `release` and `git log --oneline -5`, read the plan file, then start the
  next part (plan first if the user wants to review it).
- Workflow: implement one plan step at a time; run the headless tests,
  commit, then stop so the user can test in Slicer before the next step.

## How the cut works (Phase 1-2, in OsteotomyCutsLogic.cutPolyData)
- Sheets from buildSheetPolyData (ruled; per-point directions and depths
  already supported for Phase 3 templates). Sheets are applied in turn.
- Zero kerf: signed distance, clip at 0 (splitByDistance).
- Kerf: refineNearSheet (edges < kerf/2 near the sheet), then removeKerf
  (UNSIGNED distance |d| < kerf/2, exact root-finding clip). Sides are
  recovered afterwards from the per-sheet "SheetSide<i>" array.
- Capping (capCutFaces, per sheet, right after it): SheetParameterisation
  is one 2D chart (u, w) of the whole cut surface: + side, rounded groove
  floor, - side, with fold arcs/creases. Rim loops are triangulated there:
  scipy Delaunay of rim + graded fill points (accepted only if bounded by
  exactly the rim), fallback vtkContourTriangulator + point insertion;
  then Delaunay flips and edge splitting to CAP_TOLERANCE.
- Limited cuts (depth or options.endExtension set): with kerf 0 they use
  LIMITED_IDEAL_KERF_WIDTH (0.1 mm) through the kerf path, because the
  zero-width clip follows the signed-distance zero level, which continues
  past the sheet's edges through the whole model. A kerf sheet whose END
  rulings lie in the bone is refused (sheetEndsInModel): the chart has no
  rounded slot ends, so such slots cannot be capped yet.
- Preview outline (computeCutOutline): exact sheet/bone intersection via
  vtkStaticCellLocator.FindCellsAlongPlane per sheet triangle, plane
  intersection, Cyrus-Beck clip to the triangle.
- Fragments: connectivity, enclosed shells merged into their host,
  display normals split at sharp edges.
- Solid bone (applyCut -> getModelToCut, not in cutPolyData): with
  treatBoneAsSolid the world mesh is replaced by makeSolidPolyData (stencil +
  surface voxels, slab-wise EDT closing, cavity fill, peel of outer surface
  voxels, flying edges + windowed sinc), cached per model. Models with
  attribute OsteotomyCuts.Solid = "1" (button output, capped segments of a
  solid cut) are cut as they are.
- scipy (bundled with Slicer) is used with guarded imports.

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
- Exception (user-approved 2026-09-28): C:\Dev\Models holds de-identified
  STL models the user provided (Mandible.stl, Lower/Upper Teeth.stl,
  Mandibular canal.stl, skull models; RAS, anterior +y, superior +z). Read them IN PLACE
  only; never copy or commit them. Checks (exit 0; skipped if unset):
  $env:OSTEOTOMYCUTS_MODELS = "C:\Dev\Models"; & "C:\ProgramData\slicer.org\3D Slicer 5.13.0-2026-06-30\Slicer.exe" --no-splash --no-main-window --python-script "C:\Dev\OsteotomyCuts\OsteotomyCuts\Testing\Python\run_real_models.py" | Out-Host; $LASTEXITCODE
- Headless test run (PowerShell); exit code 0 = all passed, 1 = failure:
  & "C:\ProgramData\slicer.org\3D Slicer 5.13.0-2026-06-30\Slicer.exe" --no-splash --no-main-window --python-script "C:\Dev\OsteotomyCuts\OsteotomyCuts\Testing\Python\run_headless_tests.py" | Out-Host; $LASTEXITCODE
- Benchmark of the cutting core (~200k-triangle synthetic mesh; timings per
  stage; exit code 0 = all capped cuts watertight):
  & "C:\ProgramData\slicer.org\3D Slicer 5.13.0-2026-06-30\Slicer.exe" --no-splash --no-main-window --python-script "C:\Dev\OsteotomyCuts\OsteotomyCuts\Testing\Python\run_benchmark.py" | Out-Host; $LASTEXITCODE
- Do not use slicer.util.selectModule() headlessly: it needs a main window,
  raises, and Slicer then never exits.

## Style
- British English in UI text and comments.
- PEP 8, type hints, docstrings on all Logic methods.
- Small, focused commits with clear messages.
- Before large changes, propose a plan and wait for approval.