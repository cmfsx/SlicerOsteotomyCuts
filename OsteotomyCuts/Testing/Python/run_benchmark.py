"""Headless benchmark of the Osteotomy Cuts cutting core on a mesh the size of a segmented jaw.

Run with Slicer's --python-script option (as run_headless_tests.py). Synthetic geometry only:
a sphere of radius 40 mm with about 200 000 triangles (0.8 mm edges, like a bone surface
segmented from CT). Prints the time of each cut, split by stage, and the result. Exits 0 when
every cut gives watertight fragments, 1 otherwise.
"""

import time
import traceback
from collections import defaultdict

import numpy as np
import slicer
import vtk

exitCode = 1
try:
    import OsteotomyCuts
    from OsteotomyCuts import CutOptions, OsteotomyCutsLogic

    STAGES = ("labelEnclosedComponents", "computeSheetDistance", "refineNearSheet", "splitByDistance",
              "removeKerf", "capCutFaces", "extractFragments", "mergeEnclosedPieces", "computeDisplayNormals")

    def timedLogic(timings: dict) -> OsteotomyCutsLogic:
        """A logic whose stage methods add their run time to timings."""
        logic = OsteotomyCutsLogic()
        for name in STAGES:
            method = getattr(logic, name)

            def timed(*args, _method=method, _name=name, **kwargs):
                start = time.perf_counter()
                try:
                    return _method(*args, **kwargs)
                finally:
                    timings[_name] += time.perf_counter() - start
            setattr(logic, name, timed)
        return logic

    def openEdgeCount(polyData: vtk.vtkPolyData) -> int:
        edges = vtk.vtkFeatureEdges()
        edges.SetInputData(OsteotomyCutsLogic().mergeCoincidentPoints(polyData))
        edges.BoundaryEdgesOn()
        edges.NonManifoldEdgesOn()
        edges.FeatureEdgesOff()
        edges.ManifoldEdgesOff()
        edges.Update()
        return edges.GetOutput().GetNumberOfCells()

    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetRadius(40.0)
    sphereSource.SetThetaResolution(320)
    sphereSource.SetPhiResolution(320)
    sphereSource.Update()
    model = sphereSource.GetOutput()
    print(f"Model: {model.GetNumberOfPoints()} points, {model.GetNumberOfPolys()} triangles")

    down = np.array([0.0, 0.0, -1.0])
    straight = np.array([[1.3, -60.0, 39.0], [1.3, 60.0, 39.0]])
    across = np.array([[-60.0, 2.7, 39.0], [60.0, 2.7, 39.0]])
    stepped = np.array([[-60.0, -10.3, 39.0], [5.3, -10.3, 39.0], [5.3, 20.3, 39.0], [60.0, 20.3, 39.0]])

    def options(kerf: float = 0.0, depth: float = 0.0, cap: bool = True) -> CutOptions:
        result = CutOptions()
        result.kerfWidth, result.depth, result.capCutFaces = kerf, depth, cap
        return result

    cases = [
        ("zero kerf, through, capped", [(straight, None)], options()),
        ("kerf 1.0, through, uncapped", [(straight, None)], options(1.0, cap=False)),
        ("kerf 1.0, through, capped", [(straight, None)], options(1.0)),
        ("kerf 1.0, groove 8 mm, capped", [(straight, 8.0)], options(1.0, 8.0)),
        ("kerf 1.0, stepped groove 8 mm, capped", [(stepped, 8.0)], options(1.0, 8.0)),
        ("kerf 1.0, groove + crossing through-cut", [(across, 8.0), (straight, None)], options(1.0)),
        ("kerf 1.0, two crossing through-cuts", [(across, None), (straight, None)], options(1.0)),
    ]
    extent = OsteotomyCutsLogic().computeAutoExtent(model)
    allWatertight = True
    for name, paths, cutOptions in cases:
        timings = defaultdict(float)
        logic = timedLogic(timings)
        sheets = [logic.buildSheetPolyData(path, down, extent, depth=depth) for path, depth in paths]
        start = time.perf_counter()
        fragments = logic.cutPolyData(model, sheets, cutOptions)
        total = time.perf_counter() - start
        openEdges = [openEdgeCount(fragment) for fragment in fragments]
        watertight = all(count == 0 for count in openEdges)
        if cutOptions.capCutFaces:
            allWatertight = allWatertight and watertight
        points = sum(fragment.GetNumberOfPoints() for fragment in fragments)
        print(f"\n{name}: {total:.2f} s, {len(fragments)} fragment(s), {points} points, "
              f"{'watertight' if watertight else f'open edges {openEdges}'}")
        for stage in STAGES:
            if timings[stage] > 0:
                print(f"    {stage:<26} {timings[stage]:6.2f} s")
    print("\nBENCHMARK " + ("PASSED" if allWatertight else "FAILED: a capped cut is not watertight"))
    exitCode = 0 if allWatertight else 1
except Exception:
    traceback.print_exc()
finally:
    slicer.util.exit(exitCode)
