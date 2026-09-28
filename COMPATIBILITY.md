# Compatibility report (v1.0.0)

Osteotomy Cuts 1.0.0 was developed on **3D Slicer 5.13.0 (preview, revision 34833, built 2026-06-30)** and verified on **3D Slicer 5.12.4 (stable, revision 34645, built 2026-09-09)**, both on Windows 11: all 87 headless tests pass on both. Not yet tested on macOS or Linux. Minimum expected: Slicer 5.8 (Python 3.12, VTK 9, Qt 5). Re-check the items below on each new stable release and on other platforms.

## APIs to verify

| Area | Used for | Risk | Guard / note |
|---|---|---|---|
| `slicer.parameterNodeWrapper`: nested `parameterPack` (`CutOptions`), `Annotated[..., WithinRange]`, enums, node references, `tuple[float, float, float]` | module parameters and GUI binding | medium: behaviour changed between releases | Old scenes are upgraded by `addMissingParameters`. `Optional[tuple]` is avoided (fails in 5.13). Structures to protect and line settings are stored as node attributes, not as `list[parameterPack]`. |
| SciPy (`scipy.ndimage`, `scipy.spatial`, `scipy.sparse`) | solid bone models, capping, distances | low: bundled with Slicer | Guarded imports: without SciPy the solid option is disabled and capping falls back to VTK. |
| `vtkContourTriangulator.TriangulateContours` | fallback capping | low | Used only when the Delaunay capping fails. |
| NumPy 1.x vs 2.x (`np.unique(..., return_inverse=True)` shape) | mesh bookkeeping | low | Results are always `ravel()`ed or reshaped explicitly. |
| Python ≥ 3.10 syntax (`X \| Y` in annotations, `match` not used) | type hints evaluated at import | high on Slicer ≤ 5.6 (Python 3.9) | Requires Slicer ≥ 5.8. |
| `vtkPolyDataToImageStencil`, `vtkImageStencilToImage`, `vtkFlyingEdges3D`, `vtkWindowedSincPolyDataFilter` | solid bone models | low | Long-standing VTK filters. |
| `vtkStaticCellLocator.FindCellsAlongPlane` | red cut outline in the preview | medium: VTK ≥ 9.1 | |
| `vtkImplicitPolyDataDistance.FunctionValue(vtkDataArray, vtkDataArray)` | signed distances, many points per call | low | |
| `vtkCellArray.SetData(offsets, connectivity)` | building meshes from NumPy | low: VTK 9 cell array API | |
| `vtkSelectEnclosedPoints` (filter form, `SelectedPoints` array) | structures to protect, inside tests | low | |
| Markups: `AddControlPointWorld(vtkVector3d)`, `SetNthControlPointLocked` | closest-point markers | low | |
| `qMRMLCheckableNodeComboBox.setCheckState(node, state)`, `ctkDoubleSpinBox.specialValueText`, `QHeaderView.setSectionResizeMode` | GUI | low | |
| `slicer.util.tryWithErrorDisplay(show=...)` | silent live distance checks | low | |
| `subprocess` with `CREATE_NO_WINDOW` | git commit recorded in provenance | low | Missing git gives "unknown". |

## Tests to run on the stable release

1. **Headless self-test** (exit code 0; about 90 s):

   ```
   Slicer --no-splash --no-main-window --python-script OsteotomyCuts/Testing/Python/run_headless_tests.py
   ```

2. **Benchmark of the cutting core** (exit code 0 when every capped cut is watertight):

   ```
   Slicer --no-splash --no-main-window --python-script OsteotomyCuts/Testing/Python/run_benchmark.py
   ```

3. **Own de-identified models** (optional; set `OSTEOTOMYCUTS_MODELS` to a folder with `Mandible.stl`, `Mandibular canal.stl`, `Lower Teeth.stl`):

   ```
   Slicer --no-splash --no-main-window --python-script OsteotomyCuts/Testing/Python/run_real_models.py
   ```

4. **Generic Slicer tests** of a build with `BUILD_TESTING` (`ctest -R OsteotomyCuts`).
5. **In the GUI:** the first-use terms dialog; *Create solid bone model*; a single-line cut with blade presets; a two-line Le Fort I; structures to protect with *Check distances*; saving and reloading the scene.
