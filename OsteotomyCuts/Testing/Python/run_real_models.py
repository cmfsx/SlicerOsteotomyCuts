"""Checks on your own STL bone models, read in place (never copied).

Run with Slicer's --python-script option (as run_headless_tests.py), with the environment
variable OSTEOTOMYCUTS_MODELS set to a folder holding "Mandible_solid.stl" (a mandible, in RAS:
anterior +y, superior +z, symphysis near x = 0). Without it the script exits 0 after saying so.

Checks: a midline split of the symphysis and a vertical cut through the left body each give
watertight bone segments. Prints the timings. Exits 0 when all checks pass, 1 otherwise.
"""

import os
import time
import traceback

import numpy as np
import slicer
import vtk

FOLDER = os.environ.get("OSTEOTOMYCUTS_MODELS", "")
exitCode = 1
try:
    from OsteotomyCuts import CutOptions, OsteotomyCutsLogic

    path = os.path.join(FOLDER, "Mandible_solid.stl")
    if not FOLDER or not os.path.exists(path):
        print("Set OSTEOTOMYCUTS_MODELS to a folder holding Mandible_solid.stl: skipped.")
        exitCode = 0
        raise SystemExit

    def openEdgeCount(polyData: vtk.vtkPolyData) -> int:
        edges = vtk.vtkFeatureEdges()
        edges.SetInputData(OsteotomyCutsLogic().mergeCoincidentPoints(polyData))
        edges.BoundaryEdgesOn()
        edges.NonManifoldEdgesOn()
        edges.FeatureEdgesOff()
        edges.ManifoldEdgesOff()
        edges.Update()
        return edges.GetOutput().GetNumberOfCells()

    mandible = slicer.util.loadModel(path)
    logic = OsteotomyCutsLogic()
    bone = logic.getWorldPolyData(mandible)
    print(f"Mandible: {bone.GetNumberOfPolys()} triangles")
    down = np.array([0.0, 0.0, -1.0])
    options = CutOptions()
    options.kerfWidth = 1.0
    failures = []
    for label, line, pieces in (("Midline split", [[0.0, 90.0, 40.0], [0.0, 30.0, 40.0]], 2),
                                ("Left body", [[-27.0, 60.0, 50.0], [-27.0, 20.0, 50.0]], 2)):
        sheet = logic.buildSheetPolyData(np.array(line), down, logic.computeAutoExtent(bone))
        start = time.perf_counter()
        segments = logic.cutPolyData(bone, [sheet], options)
        openEdges = [openEdgeCount(segment) for segment in segments]
        print(f"{label}: {time.perf_counter() - start:.2f} s, {len(segments)} bone segment(s), open edges {openEdges}")
        if len(segments) < pieces or any(openEdges):
            failures.append(f"{label}: expected at least {pieces} closed bone segments")
    print("\nREAL MODELS " + ("PASSED" if not failures else "FAILED: " + "; ".join(failures)))
    exitCode = 0 if not failures else 1
except SystemExit:
    pass
except Exception:
    traceback.print_exc()
finally:
    slicer.util.exit(exitCode)
