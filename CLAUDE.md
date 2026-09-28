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

## Current status (updated 2026-09-27)
- Phase 1: complete.
- Phase 2: complete (steps 1-5) and confirmed by the user in Slicer on
  2026-09-27 (grooves, folded/closed grooves, through-cut across a groove,
  real bone).
- v1.0.0 pre-release (branch `release`, from 3a7458d = Phase 2, no Phase 3):
  Parts 0-10 of the user's to-do list, done in order, stopping after each
  for testing in Slicer. Phase 3 work continues on `master` and is merged
  after the release (Part 10). Nothing is pushed publicly, tagged or put
  into a pull request until the user types exactly: GO RELEASE.
  Part 0 done: Phase 2 robustness fixes ported from master (cap regions by
  rim-edge direction, missing rim edges recovered by flips, cap refinement
  point budget, integer-key open edges, scene-close observer guard).
  Part 0b done (added 2026-09-28: a Le Fort I cut went through the whole
  skull): limited cuts with the ideal blade, "Past line ends"
  (options.endExtension), red cut outline in the preview, Le Fort I help.
  Confirmed by the user in Slicer on 2026-09-28 (mandible; a skull that
  did not separate turned out to be a non-solid model, which Part 1 fixes).
  Part 1 done 2026-09-28 (solid bone models: makeSolidPolyData, "Create
  solid bone model", treatBoneAsSolid on by default), AWAITING the user's
  Slicer test. Deviation from the plan: all pieces >= 1% of the largest are
  kept (not only the largest), so separate bones / both canals survive.
  Part 0d added and done 2026-09-28 (user: needed for release 1): several
  cut paths cut together as one osteotomy ("Further lines", node refs
  OsteotomyCuts.GroupLine on the first line); each later line only cuts on
  its own side of earlier lines (stops at them); per-line settings stored
  on each curve (attribute OsteotomyCuts.LineSettings + DirectionLine ref).
  Tried by the user on a skull (Le Fort I works; a new line inherited the
  first line's depth -> fixed: new lines start with depth/reach/direction
  reset). Release 1 must have (user, 2026-09-28): solid bone, multi-line
  osteotomy, capping check + warning, surgeon wording, nerve/teeth warnings.
  Also done: small loose pieces joined to a neighbouring segment
  (minSegmentPercent, default 1%); Part 2 (assessModel / assessSegments,
  pre-cut dialog, not-closed warnings); Part 3 (structures to protect:
  settings as attribute OsteotomyCuts.ProtectedStructure on the model/curve
  node instead of a parameter-node list; checkClearances samples each
  line's sheet inside the bone, 1 mm then 0.1 mm near the minimum; table,
  closest-point markups, sheet colour, pre-cut confirm, SafetyResults /
  SafetyOverride on segments). SAFETY_NOTE wording is a placeholder: ask
  the user for their exact required sentence (Part 5/6).
  NEXT: Part 4 (provenance), then Part 5 (surgeon wording), 6-10. The full
  approved plan (Parts 0-10, with details per part) is in the user's
  Claude plans folder: twinkly-brewing-phoenix.md.
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
  STL models the user provided (Mandible_solid.stl, Lower Teeth.stl,
  Mandibular canal.stl; RAS, anterior +y, superior +z). Read them IN PLACE
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