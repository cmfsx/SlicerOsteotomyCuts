# SPDX-FileCopyrightText: 2026 Manjula Herath
# SPDX-License-Identifier: GPL-3.0-or-later
# Part of OsteotomyCuts, a 3D Slicer extension. See LICENSE and DISCLAIMER.md.

"""Headless benchmark of the Osteotomy Cuts cutting core on a mesh the size of a segmented jaw.

Run with Slicer's --python-script option (as run_headless_tests.py). Synthetic geometry only:
a sphere of radius 40 mm with about 200 000 triangles (0.8 mm edges, like a bone surface
segmented from CT), and a jaw-like segmentation mesh of about a million triangles with a bumpy
surface and marrow cavities. Prints the time of each cut, split by stage, and the result.
Exits 0 when every capped cut gives watertight fragments, 1 otherwise.
"""

import time
import traceback
from collections import defaultdict

import numpy as np
import slicer
import vtk
from vtk.util import numpy_support

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

    def syntheticJaw() -> vtk.vtkPolyData:
        """A jaw-like bone as a segmentation export gives it: marching cubes (0.25 mm voxels) of
        a thick U-shaped bar with two rami, with a bumpy surface and marrow cavities, smoothed."""
        from scipy import ndimage
        spacing = 0.25
        lower, upper = np.array([-58.0, -14.0, -20.0]), np.array([58.0, 82.0, 58.0])
        shape = np.ceil((upper - lower) / spacing).astype(int) + 1
        x = np.linspace(-40.0, 40.0, 400)
        centre = [np.column_stack([x, 0.022 * x ** 2, np.zeros_like(x)])]
        for side in (-1.0, 1.0):
            t = np.linspace(0.0, 1.0, 200)[:, np.newaxis]
            centre.append(np.array([side * 40.0, 35.2, 0.0]) + t * np.array([side * 2.0, 20.0, 42.0]))
        centre = np.vstack(centre)
        dense = np.vstack([np.linspace(centre[i], centre[i + 1], 6) for i in range(len(centre) - 1)])
        index = np.clip(np.rint((dense - lower) / spacing).astype(int), 0, shape - 1)
        outside = np.ones(shape, dtype=bool)
        outside[index[:, 0], index[:, 1], index[:, 2]] = False
        distance = ndimage.distance_transform_edt(outside) * spacing
        noise = ndimage.gaussian_filter(np.random.default_rng(1).standard_normal(shape).astype(np.float32), 4.0)
        field = (distance - 6.0 + noise * (1.6 / noise.std())).astype(np.float32)  # < 0 in the bone
        image = vtk.vtkImageData()
        image.SetDimensions(*shape.tolist())
        image.SetSpacing(spacing, spacing, spacing)
        image.SetOrigin(*lower.tolist())
        image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(field.ravel(order="F"), deep=True))
        contour = vtk.vtkFlyingEdges3D()
        contour.SetInputData(image)
        contour.SetValue(0, 0.0)
        smooth = vtk.vtkWindowedSincPolyDataFilter()
        smooth.SetInputConnection(contour.GetOutputPort())
        smooth.SetNumberOfIterations(15)
        smooth.SetPassBand(0.1)
        smooth.NormalizeCoordinatesOn()
        smooth.Update()
        return smooth.GetOutput()

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

    # A segmentation-like jaw: about a million triangles, a bumpy surface and marrow cavities.
    # Guards against cut faces that fail to close or never finish refining on real bone.
    jaw = syntheticJaw()
    timings = defaultdict(float)
    logic = timedLogic(timings)
    leftBody = np.array([[-30.0, 0.0, 20.0], [-30.0, 40.0, 20.0]])  # across the left body
    sheet = logic.buildSheetPolyData(leftBody, down, logic.computeAutoExtent(jaw))
    start = time.perf_counter()
    fragments = logic.cutPolyData(jaw, [sheet], options(1.0))
    total = time.perf_counter() - start
    openEdges = [openEdgeCount(fragment) for fragment in fragments]
    jawWatertight = all(count == 0 for count in openEdges)
    allWatertight = allWatertight and jawWatertight
    print(f"\nSynthetic jaw ({jaw.GetNumberOfPolys()} triangles), kerf 1.0 through the left body: {total:.2f} s, "
          f"{len(fragments)} fragment(s), {'watertight' if jawWatertight else f'open edges {openEdges}'}")
    for stage in STAGES:
        if timings[stage] > 0:
            print(f"    {stage:<26} {timings[stage]:6.2f} s")
    print("\nBENCHMARK " + ("PASSED" if allWatertight else "FAILED: a capped cut is not watertight"))
    exitCode = 0 if allWatertight else 1
except Exception:
    traceback.print_exc()
finally:
    slicer.util.exit(exitCode)
