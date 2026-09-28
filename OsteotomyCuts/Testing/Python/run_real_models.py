# SPDX-FileCopyrightText: 2026 Manjula Herath
# SPDX-License-Identifier: GPL-3.0-or-later
# Part of OsteotomyCuts, a 3D Slicer extension. See LICENSE and DISCLAIMER.md.

"""Checks on your own STL bone models, read in place (never copied).

Run with Slicer's --python-script option (as run_headless_tests.py), with the environment
variable OSTEOTOMYCUTS_MODELS set to a folder holding "Mandible_solid.stl" (a mandible, in RAS:
anterior +y, superior +z, symphysis near x = 0). Without it the script exits 0 after saying so.

Checks: a midline split of the symphysis and a vertical cut through the left body each give
watertight bone segments; a groove; a solid version of the mandible is closed, keeps its volume
and splits into closed segments; the cut outline is quick. Prints the timings. Exits 0 when all checks pass, 1 otherwise.
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

    path = next((os.path.join(FOLDER, name) for name in ("Mandible_solid.stl", "Mandible.stl")
                 if FOLDER and os.path.exists(os.path.join(FOLDER, name))), "")
    if not path:
        print("Set OSTEOTOMYCUTS_MODELS to a folder holding Mandible.stl: skipped.")
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

    # A limited cut with the ideal blade (thin numerical layer): a groove, one closed segment
    limited = CutOptions()
    limited.depth = 8.0
    surfaceLocator = vtk.vtkCellLocator()
    surfaceLocator.SetDataSet(bone)
    surfaceLocator.BuildLocator()
    path = []
    for y in (40.0, 20.0):  # on the bone surface, as snapping places them: topmost hit from above
        hits, cellIds = vtk.vtkPoints(), vtk.vtkIdList()
        surfaceLocator.IntersectWithLine([-27.0, y, 200.0], [-27.0, y, -200.0], 0.0, hits, cellIds)
        points = np.array([hits.GetPoint(i) for i in range(hits.GetNumberOfPoints())])
        path.append(points[np.argmax(points[:, 2])])
    sheet = logic.buildSheetPolyData(np.array(path), down, logic.computeAutoExtent(bone), depth=limited.depth)
    start = time.perf_counter()
    segments = logic.cutPolyData(bone, [sheet], limited)
    openEdges = [openEdgeCount(segment) for segment in segments]
    print(f"Left body groove 8 mm, ideal blade: {time.perf_counter() - start:.2f} s, "
          f"{len(segments)} bone segment(s), open edges {openEdges}")
    if len(segments) < 1 or any(openEdges):
        failures.append("Left body groove: expected closed bone segments")

    # Solid bone model (release Part 1): closed, one piece, close to the (already solid) mandible,
    # and a cut of it gives closed bone segments
    start = time.perf_counter()
    solid = logic.makeSolidPolyData(bone)
    elapsed = time.perf_counter() - start
    volumes = []
    for mesh in (bone, solid):
        massProperties = vtk.vtkMassProperties()
        massProperties.SetInputData(mesh)
        massProperties.Update()
        volumes.append(massProperties.GetVolume())
    change = volumes[1] / volumes[0] - 1.0
    print(f"Solid mandible: {elapsed:.1f} s, {solid.GetNumberOfPolys()} triangles, open edges {openEdgeCount(solid)}, "
          f"volume change {100.0 * change:+.2f}%")
    if openEdgeCount(solid) or abs(change) > 0.02:
        failures.append("Solid mandible: expected a closed model within 2% of the volume")
    solidSheet = logic.buildSheetPolyData(np.array([[0.0, 90.0, 40.0], [0.0, 30.0, 40.0]]), down,
                                          logic.computeAutoExtent(solid))
    start = time.perf_counter()
    segments = logic.cutPolyData(solid, [solidSheet], options)
    openEdges = [openEdgeCount(segment) for segment in segments]
    print(f"Solid mandible midline split: {time.perf_counter() - start:.2f} s, {len(segments)} bone segment(s), "
          f"open edges {openEdges}")
    if len(segments) != 2 or any(openEdges):
        failures.append("Solid mandible midline split: expected 2 closed bone segments")

    # Structures to protect (release Part 3): a vertical cut through the left body crosses the
    # mandibular canal; the teeth are checked too. Uses the solid mandible for the inside test.
    from OsteotomyCuts import ClearanceStatus, OsteotomyLine
    canalPath = os.path.join(FOLDER, "Mandibular canal.stl")
    if os.path.exists(canalPath):
        structures = []
        for name in ("Mandibular canal.stl", "Lower Teeth.stl"):
            if os.path.exists(os.path.join(FOLDER, name)):
                structures.append(logic.addProtectedStructure(slicer.util.loadModel(os.path.join(FOLDER, name))))
        bodyLine = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", "LeftBody")
        bodySheet = logic.buildSheetPolyData(np.array([[-27.0, 60.0, 50.0], [-27.0, 20.0, 50.0]]), down,
                                             logic.computeAutoExtent(solid))
        start = time.perf_counter()
        results = logic.checkClearances([OsteotomyLine(bodyLine, bodySheet, options, [])], solid, structures)
        elapsed = time.perf_counter() - start
        for result in results:
            print(f"Clearance {result.structure.node.GetName()}: {result.clearance:.2f} mm, {result.status.value}")
        print(f"Clearance check: {elapsed:.2f} s")
        canal = next(r for r in results if r.structure.node.GetName().startswith("Mandibular canal"))
        if canal.status != ClearanceStatus.ENTERS or elapsed > 10.0:
            failures.append("Clearance: expected the body cut to enter the canal, within 10 s")

    # Cut outline for the live preview: must be quick on a real bone
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(bone)
    start = time.perf_counter()
    locator.BuildLocator()
    built = time.perf_counter() - start
    start = time.perf_counter()
    outline = logic.computeCutOutline(bone, sheet, locator)
    elapsed = time.perf_counter() - start
    print(f"Cut outline: locator {built:.2f} s once, then {elapsed * 1000:.0f} ms, {outline.GetNumberOfLines()} segments")
    if outline.GetNumberOfLines() == 0 or elapsed > 0.5:
        failures.append("Cut outline: expected lines within 0.5 s")
    print("\nREAL MODELS " + ("PASSED" if not failures else "FAILED: " + "; ".join(failures)))
    exitCode = 0 if not failures else 1
except SystemExit:
    pass
except Exception:
    traceback.print_exc()
finally:
    slicer.util.exit(exitCode)
