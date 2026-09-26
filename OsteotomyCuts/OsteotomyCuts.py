import enum
from dataclasses import dataclass
from typing import Annotated, Optional

import numpy as np
import vtk
from vtk.util import numpy_support

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    parameterPack,
    WithinRange,
)

from slicer import (
    vtkMRMLMarkupsCurveNode,
    vtkMRMLMarkupsLineNode,
    vtkMRMLModelNode,
)


#
# OsteotomyCuts
#


class OsteotomyCuts(ScriptedLoadableModule):
    """Virtual osteotomy: multi-segment cuts of bone models for surgical planning."""

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Osteotomy Cuts")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Surgical planning")]
        self.parent.dependencies = ["Markups", "Models", "SubjectHierarchy"]
        self.parent.contributors = ["Manjula Herath (FaceLab.care)"]
        self.parent.helpText = _("""
Virtual osteotomy for orthognathic and craniofacial surgical planning.
Place the points of a cut path on the bone surface, choose an extrusion direction, and
the bone model is split along the extruded cutting sheet into separate fragments.
The original model is hidden, never modified.
""")
        self.parent.acknowledgementText = _("""
Based on the 3D Slicer scripted module template developed by Jean-Christophe Fillion-Robin,
Kitware Inc., Andras Lasso, PerkLab, and Steve Pieper, Isomics, Inc.
""")


#
# Parameter types
#


class DirectionMode(enum.Enum):
    """Source of the extrusion direction of the cutting sheet."""

    VIEW = "view"  # direction captured from a 3D view camera (stored, not followed live)
    LINE = "line"  # direction of a markups line (point 0 -> point 1)


NOT_CAPTURED = (0.0, 0.0, 0.0)


def isDirectionCaptured(direction: tuple[float, float, float]) -> bool:
    """Return True if a stored direction holds a usable (non-zero) vector."""
    return any(abs(component) > 1e-9 for component in direction)


@parameterPack
class CutOptions:
    """Options for one cut. Phase 2 adds kerf width, depth and capping here."""

    # Distance (mm) the sheet extends past the path; 0 = automatic (bounding-box diagonal)
    extension: Annotated[float, WithinRange(0.0, 10000.0)] = 0.0
    # Free-standing pieces smaller than this fraction of the model's points are discarded
    minFragmentFraction: Annotated[float, WithinRange(0.0, 0.5)] = 0.001


@dataclass
class FragmentPiece:
    """One connected piece of a cut model, before enclosed pieces are merged into their host."""

    polyData: vtk.vtkPolyData
    componentId: int  # connected component of the uncut model the piece came from
    hostComponentId: int  # component enclosing that component, or -1 if free-standing
    sideSignature: tuple[int, ...]  # +1 / -1 side of each cutting sheet
    pointCount: int


#
# OsteotomyCutsParameterNode
#


@parameterNodeWrapper
class OsteotomyCutsParameterNode:
    """
    The parameters needed by the module.

    inputModel - Model to cut (a bone, or a fragment from an earlier cut).
    cutCurve - Markups curve along the osteotomy, points on the model surface.
    directionMode - Whether the extrusion direction comes from a 3D view or a markups line.
    viewDirection - Captured 3D view direction (RAS unit vector); NOT_CAPTURED until captured.
    directionLine - Markups line defining the extrusion direction in LINE mode.
    snapToSurface - Keep the cut path points on the model surface.
    livePreview - Show and update the cutting sheet while points are moved.
    options - Cut options.
    sheetModel - Model node showing the cutting sheet preview.
    """

    inputModel: vtkMRMLModelNode
    cutCurve: vtkMRMLMarkupsCurveNode
    directionMode: DirectionMode = DirectionMode.VIEW
    # A zero vector marks "not captured". Optional[tuple] cannot be used: in Slicer 5.13 the
    # union serialiser cannot match a tuple value to the tuple serialiser, so writes fail.
    viewDirection: tuple[float, float, float] = NOT_CAPTURED
    directionLine: vtkMRMLMarkupsLineNode
    snapToSurface: bool = True
    livePreview: bool = True
    options: CutOptions
    sheetModel: vtkMRMLModelNode


#
# OsteotomyCutsWidget
#


class OsteotomyCutsWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialised."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialised."""
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/OsteotomyCuts.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = OsteotomyCutsLogic()

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # The direction mode enum has no ready-made radio button connector, so it is wired here
        self.ui.directionViewRadioButton.connect("toggled(bool)", self.onDirectionModeToggled)
        self.ui.directionLineRadioButton.connect("toggled(bool)", self.onDirectionModeToggled)

        # Make sure parameter node is initialised (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateGuiFromParameterNode)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode: Optional[OsteotomyCutsParameterNode]) -> None:
        """Set and observe parameter node, so that the GUI follows parameter changes."""
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateGuiFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Widgets with a "SlicerParameterName" property in the .ui file are connected here
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateGuiFromParameterNode)
            self._updateGuiFromParameterNode()

    def onDirectionModeToggled(self, checked: bool) -> None:
        """Store the direction mode selected with the radio buttons."""
        if not self._parameterNode or not checked:
            return
        if self.ui.directionLineRadioButton.checked:
            self._parameterNode.directionMode = DirectionMode.LINE
        else:
            self._parameterNode.directionMode = DirectionMode.VIEW

    def _updateGuiFromParameterNode(self, caller=None, event=None) -> None:
        """Update the widgets that are not connected automatically by the parameter node wrapper."""
        if not self._parameterNode:
            return

        isLineMode = self._parameterNode.directionMode == DirectionMode.LINE
        for radioButton, checked in ((self.ui.directionLineRadioButton, isLineMode),
                                     (self.ui.directionViewRadioButton, not isLineMode)):
            wasBlocked = radioButton.blockSignals(True)
            radioButton.checked = checked
            radioButton.blockSignals(wasBlocked)
        self.ui.captureViewDirectionButton.enabled = not isLineMode
        self.ui.viewDirectionLabel.enabled = not isLineMode
        self.ui.directionLineSelector.enabled = isLineMode
        self.ui.directionLinePlaceWidget.enabled = isLineMode

        viewDirection = self._parameterNode.viewDirection
        if not isDirectionCaptured(viewDirection):
            self.ui.viewDirectionLabel.text = _("Not captured")
        else:
            self.ui.viewDirectionLabel.text = "({:.2f}, {:.2f}, {:.2f})".format(*viewDirection)

        # Cutting is added in later implementation steps
        self.ui.applyButton.enabled = False
        self.ui.undoButton.enabled = False
        self.ui.mergeButton.enabled = False
        self.ui.statusLabel.text = _("Cutting is not implemented yet.")


#
# OsteotomyCutsLogic
#


class OsteotomyCutsLogic(ScriptedLoadableModuleLogic):
    """Computation for the Osteotomy Cuts module. Runs without the GUI (headless)."""

    def __init__(self) -> None:
        """Initialise the logic."""
        ScriptedLoadableModuleLogic.__init__(self)

    # A path segment closer than this angle to its extrusion direction is rejected, because the
    # sheet would degenerate to a sliver there.
    MIN_SEGMENT_DIRECTION_ANGLE_DEG = 5.0

    # Consecutive path points closer than this (mm) are treated as duplicates
    DUPLICATE_POINT_TOLERANCE = 1e-6

    def getParameterNode(self) -> OsteotomyCutsParameterNode:
        """Return the module's parameter node, creating it if needed."""
        return OsteotomyCutsParameterNode(super().getParameterNode())

    #
    # Geometry (no MRML)
    #

    def buildSheetPolyData(self, pathPoints: np.ndarray, directions: np.ndarray,
                           extent: float, closed: bool = False) -> vtk.vtkPolyData:
        """Build a ruled cutting sheet by extruding a polyline.

        Each path point P is extruded to A = P - extent * d and B = P + extent * d, so the sheet
        reaches ``extent`` to both sides of the path. An open path is also extended by
        ``extent`` at both ends, along the end tangent with its component along d removed,
        so that the sheet edges lie outside the model.

        Output point order is A, B pairs along the (extended) path:
        ``[A_start, B_start, A_0, B_0, ..., A_last, B_last, A_end, B_end]`` for an open path and
        ``[A_0, B_0, ..., A_last, B_last]`` for a closed path.

        :param pathPoints: (N, 3) world points along the cut path, N >= 2 (N >= 3 if closed).
            Consecutive duplicates are removed.
        :param directions: (3,) one extrusion direction, or (N, 3) one per path point
            (Phase 3 BSSO templates use a different direction per segment). Normalised here.
        :param extent: distance (mm) the sheet reaches along +/- direction and past the ends.
        :param closed: join the last point to the first and do not extend the ends.
        :return: triangulated sheet with consistent winding, point and cell normals.
        :raises ValueError: too few distinct points, a zero or malformed direction, a
            non-positive extent, or a path segment (nearly) parallel to its direction.
        """
        points = np.asarray(pathPoints, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(_("Path points must be an (N, 3) array."))
        if not extent > 0:
            raise ValueError(_("Sheet extent must be positive."))

        dirs = np.asarray(directions, dtype=float)
        if dirs.shape == (3,):
            dirs = np.tile(dirs, (len(points), 1))
        elif dirs.shape != points.shape:
            raise ValueError(_("Directions must be a single 3-vector or one 3-vector per path point."))
        lengths = np.linalg.norm(dirs, axis=1)
        if np.any(lengths < 1e-9):
            raise ValueError(_("The extrusion direction must not be a zero vector."))
        dirs = dirs / lengths[:, np.newaxis]

        # Remove consecutive duplicates (and, for a closed path, a repeated first point at the end)
        if len(points) > 0:
            keep = np.ones(len(points), dtype=bool)
            keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > self.DUPLICATE_POINT_TOLERANCE
            points, dirs = points[keep], dirs[keep]
        if closed and len(points) > 1 and np.linalg.norm(points[-1] - points[0]) <= self.DUPLICATE_POINT_TOLERANCE:
            points, dirs = points[:-1], dirs[:-1]

        minPoints = 3 if closed else 2
        if len(points) < minPoints:
            raise ValueError(_("The cut path needs at least {count} distinct points.").format(count=minPoints))

        # Reject segments (nearly) parallel to the extrusion direction at either end
        maxCosine = np.cos(np.radians(self.MIN_SEGMENT_DIRECTION_ANGLE_DEG))
        segmentEnds = np.roll(np.arange(len(points)), -1)
        segmentCount = len(points) if closed else len(points) - 1
        for i in range(segmentCount):
            j = segmentEnds[i]
            tangent = points[j] - points[i]
            tangent /= np.linalg.norm(tangent)
            if max(abs(np.dot(tangent, dirs[i])), abs(np.dot(tangent, dirs[j]))) > maxCosine:
                raise ValueError(_("Cut path segment {index} is almost parallel to the extrusion direction.")
                                 .format(index=i + 1))

        if not closed:
            startPoint = points[0] + extent * self._endExtensionDirection(points[0] - points[1], dirs[0])
            endPoint = points[-1] + extent * self._endExtensionDirection(points[-1] - points[-2], dirs[-1])
            points = np.vstack([startPoint, points, endPoint])
            dirs = np.vstack([dirs[0], dirs, dirs[-1]])

        # Interleaved A_i, B_i vertices
        vertices = np.empty((2 * len(points), 3))
        vertices[0::2] = points - extent * dirs
        vertices[1::2] = points + extent * dirs

        vtkPoints = vtk.vtkPoints()
        vtkPoints.SetData(numpy_support.numpy_to_vtk(vertices, deep=True))

        # Two triangles per strip quad, same winding for every quad
        triangles = vtk.vtkCellArray()
        quadCount = len(points) if closed else len(points) - 1
        for i in range(quadCount):
            j = (i + 1) % len(points)
            a0, b0, a1, b1 = 2 * i, 2 * i + 1, 2 * j, 2 * j + 1
            for triangle in ((a0, a1, b1), (a0, b1, b0)):
                triangles.InsertNextCell(3)
                for pointId in triangle:
                    triangles.InsertCellPoint(pointId)

        sheet = vtk.vtkPolyData()
        sheet.SetPoints(vtkPoints)
        sheet.SetPolys(triangles)

        # Point normals without splitting, so that points stay shared at folds and the
        # signed distance sees one smooth pseudo-normal there.
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(sheet)
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOn()
        normals.SplittingOff()
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOff()
        normals.Update()

        result = vtk.vtkPolyData()
        result.DeepCopy(normals.GetOutput())
        return result

    @staticmethod
    def _endExtensionDirection(outwardTangent: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Unit vector continuing the path outward at an end, perpendicular to the direction."""
        perpendicular = outwardTangent - np.dot(outwardTangent, direction) * direction
        return perpendicular / np.linalg.norm(perpendicular)

    def computeAutoExtent(self, polyData: vtk.vtkPolyData, margin: float = 0.0) -> float:
        """Return a sheet extent that always clears the model.

        :param polyData: model mesh (world coordinates).
        :param margin: extra distance (mm) added to the bounding-box diagonal.
        :return: bounding-box diagonal of the mesh plus margin.
        :raises ValueError: if the mesh has no points.
        """
        if polyData is None or polyData.GetNumberOfPoints() == 0:
            raise ValueError(_("The model has no points."))
        bounds = np.array(polyData.GetBounds())
        diagonal = float(np.linalg.norm(bounds[1::2] - bounds[0::2]))
        return diagonal + margin

    def labelEnclosedComponents(self, polyData: vtk.vtkPolyData,
                                minHostFraction: float) -> vtk.vtkPolyData:
        """Label connected components of the uncut model and record which sit inside another.

        Internal shells such as the inferior alveolar canal or marrow voids are separate
        components enclosed by the bone's outer surface; they must travel with their segment.
        The inside test is done here, on the uncut mesh, where shells are closed.

        :param polyData: triangulated model mesh (not modified).
        :param minHostFraction: only components with at least this fraction of all points are
            tried as hosts.
        :return: copy of the mesh with integer point arrays "ComponentId" and "HostComponentId".
            HostComponentId is the smallest enclosing component, or -1 when free-standing.
        """
        labelled, pointComponents, cellComponents, componentCount = self._connectedRegions(polyData)
        components = self._splitByCellLabel(labelled, cellComponents)
        pointCounts = {c: components[c].GetNumberOfPoints() for c in components}
        bounds = {c: np.array(components[c].GetBounds()) for c in components}

        def boxVolume(b: np.ndarray) -> float:
            return float(np.prod(b[1::2] - b[0::2]))

        def boxInside(inner: np.ndarray, outer: np.ndarray) -> bool:
            return bool(np.all(inner[0::2] >= outer[0::2]) and np.all(inner[1::2] <= outer[1::2]))

        minHostPoints = max(4, minHostFraction * labelled.GetNumberOfPoints())
        hostOf = np.full(max(componentCount, 1), -1, dtype=np.int32)
        for host in (c for c in components if pointCounts[c] >= minHostPoints):
            selector = vtk.vtkSelectEnclosedPoints()
            selector.Initialize(components[host])
            for candidate in components:
                if candidate == host or not boxInside(bounds[candidate], bounds[host]):
                    continue
                if boxVolume(bounds[candidate]) >= boxVolume(bounds[host]):
                    continue  # also rules out cycles between identical shells
                currentHost = hostOf[candidate]
                if currentHost >= 0 and boxVolume(bounds[currentHost]) <= boxVolume(bounds[host]):
                    continue  # already inside a smaller host
                # Majority vote over a few points, in case one lies on a coincident surface
                candidatePoints = components[candidate].GetPoints()
                sampleIds = np.unique(np.linspace(0, candidatePoints.GetNumberOfPoints() - 1, 5).astype(int))
                insideVotes = sum(selector.IsInsideSurface(candidatePoints.GetPoint(int(i))) for i in sampleIds)
                if 2 * insideVotes > len(sampleIds):
                    hostOf[candidate] = host
            selector.Complete()

        self._addIntPointArray(labelled, "ComponentId", pointComponents)
        self._addIntPointArray(labelled, "HostComponentId", hostOf[pointComponents])
        return labelled

    def computeSheetDistance(self, polyData: vtk.vtkPolyData, sheetPolyData: vtk.vtkPolyData,
                             arrayName: str = "SheetDistance") -> vtk.vtkPolyData:
        """Return a copy of the mesh with a point array of signed distance to the sheet.

        The distance is evaluated in one C++ call (vtkImplicitFunction.FunctionValue), not per
        vertex from Python. The sign follows the sheet normals. Phase 2 kerf and depth read this
        array. The input mesh is not modified (points and cells are shared, arrays are not added
        to it).

        :param polyData: model mesh.
        :param sheetPolyData: cutting sheet from buildSheetPolyData.
        :param arrayName: name of the distance array.
        :return: mesh copy with the distance array set as active scalars.
        """
        implicitDistance = vtk.vtkImplicitPolyDataDistance()
        implicitDistance.SetInput(sheetPolyData)
        distances = vtk.vtkDoubleArray()
        implicitDistance.FunctionValue(polyData.GetPoints().GetData(), distances)
        distances.SetName(arrayName)

        result = vtk.vtkPolyData()
        result.CopyStructure(polyData)
        result.GetPointData().PassData(polyData.GetPointData())
        result.GetCellData().PassData(polyData.GetCellData())
        result.GetPointData().AddArray(distances)
        result.GetPointData().SetActiveScalars(arrayName)
        return result

    def splitByDistance(self, polyWithDistance: vtk.vtkPolyData, options: CutOptions,
                        arrayName: str = "SheetDistance") -> tuple[vtk.vtkPolyData, vtk.vtkPolyData]:
        """Split a mesh into the positive and the negative side of the cutting sheet.

        Phase 1 clips at distance 0 (no kerf). Phase 2 kerf removal replaces this, preferably
        with unsigned distance to the sheet interior (see CLAUDE.md).

        :param polyWithDistance: mesh with the distance point array (computeSheetDistance).
        :param options: cut options (unused until Phase 2 adds kerf).
        :param arrayName: name of the distance array.
        :return: (positive side, negative side), both triangulated; either may be empty.
        """
        polyWithDistance.GetPointData().SetActiveScalars(arrayName)
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(polyWithDistance)
        clipper.SetValue(0.0)
        clipper.GenerateClippedOutputOn()
        clipper.Update()
        return (self._ensureTriangles(clipper.GetOutput()),
                self._ensureTriangles(clipper.GetClippedOutput()))

    def extractFragments(self, polyData: vtk.vtkPolyData,
                         sideSignature: tuple[int, ...]) -> list[FragmentPiece]:
        """Split a mesh into its connected pieces.

        :param polyData: triangulated mesh with "ComponentId" and "HostComponentId" point arrays.
        :param sideSignature: side of each sheet this mesh lies on, copied to every piece.
        :return: one FragmentPiece per connected region.
        """
        if polyData.GetNumberOfCells() == 0:
            return []
        regions, _pointRegions, cellRegions, _regionCount = self._connectedRegions(polyData)
        pieces = []
        for piece in self._splitByCellLabel(regions, cellRegions).values():
            componentIds = numpy_support.vtk_to_numpy(piece.GetPointData().GetArray("ComponentId"))
            hostIds = numpy_support.vtk_to_numpy(piece.GetPointData().GetArray("HostComponentId"))
            # A connected piece comes from one component; the mode guards against interpolation noise
            componentId = int(np.bincount(componentIds).argmax())
            hostComponentId = int(hostIds[np.argmax(componentIds == componentId)])
            pieces.append(FragmentPiece(piece, componentId, hostComponentId, tuple(sideSignature),
                                        piece.GetNumberOfPoints()))
        return pieces

    def mergeEnclosedPieces(self, pieces: list[FragmentPiece],
                            minPointCount: float) -> list[vtk.vtkPolyData]:
        """Merge enclosed pieces into a piece of their host, and drop small free-standing pieces.

        An enclosed piece (e.g. half of the canal) joins a piece of its outermost host component
        on the same side of every sheet; if there are several, the nearest one wins. Enclosed
        pieces are never dropped for being small.

        :param pieces: pieces from extractFragments.
        :param minPointCount: free-standing pieces with fewer points are discarded.
        :return: fragments, largest first.
        """
        hostOfComponent = {piece.componentId: piece.hostComponentId for piece in pieces}

        def outermostHost(componentId: int) -> int:
            visited = set()
            while hostOfComponent.get(componentId, -1) >= 0 and componentId not in visited:
                visited.add(componentId)
                componentId = hostOfComponent[componentId]
            return componentId

        kept = [p for p in pieces if p.hostComponentId < 0 and p.pointCount >= minPointCount]
        groups = [[p.polyData] for p in kept]
        distanceFunctions = {}

        def distanceToKept(index: int, piece: FragmentPiece) -> float:
            if index not in distanceFunctions:
                distanceFunctions[index] = vtk.vtkImplicitPolyDataDistance()
                distanceFunctions[index].SetInput(kept[index].polyData)
            points = piece.polyData.GetPoints()
            sampleIds = np.unique(np.linspace(0, points.GetNumberOfPoints() - 1, 10).astype(int))
            return min(abs(distanceFunctions[index].EvaluateFunction(points.GetPoint(int(i)))) for i in sampleIds)

        orphans = []
        for piece in (p for p in pieces if p.hostComponentId >= 0):
            root = outermostHost(piece.componentId)
            sameComponent = [i for i, p in enumerate(kept) if p.componentId == root]
            candidates = ([i for i in sameComponent if kept[i].sideSignature == piece.sideSignature]
                          or sameComponent or list(range(len(kept))))
            if not candidates:
                orphans.append(piece.polyData)  # nothing to merge into: keep as its own fragment
                continue
            best = min(candidates, key=lambda i: distanceToKept(i, piece))
            groups[best].append(piece.polyData)

        fragments = [group[0] if len(group) == 1 else self.mergePolyData(group) for group in groups]
        fragments.extend(orphans)
        fragments.sort(key=lambda pd: pd.GetNumberOfPoints(), reverse=True)
        return fragments

    def cutPolyData(self, polyData: vtk.vtkPolyData, sheets: list[vtk.vtkPolyData],
                    options: CutOptions) -> list[vtk.vtkPolyData]:
        """Cut a mesh with one or more sheets into separate fragments (headless core).

        Each sheet is applied in turn to every current piece, then the pieces are split into
        connected fragments and enclosed pieces are merged into their host. Several sheets let
        Phase 3 cut several osteotomies at once. The input mesh is not modified.

        :param polyData: model mesh in world coordinates.
        :param sheets: cutting sheets from buildSheetPolyData.
        :param options: cut options.
        :return: fragment meshes, largest first.
        :raises ValueError: if there is no sheet or the mesh has no polygons.
        """
        if not sheets:
            raise ValueError(_("At least one cutting sheet is needed."))
        triangles = self._ensureTriangles(polyData)
        if triangles.GetNumberOfCells() == 0:
            raise ValueError(_("The model has no surface polygons to cut."))

        labelled = self.labelEnclosedComponents(triangles, options.minFragmentFraction)
        sides = [(labelled, ())]
        for sheet in sheets:
            nextSides = []
            for mesh, signature in sides:
                positive, negative = self.splitByDistance(self.computeSheetDistance(mesh, sheet), options)
                for part, side in ((positive, 1), (negative, -1)):
                    if part.GetNumberOfCells() > 0:
                        nextSides.append((part, signature + (side,)))
            sides = nextSides

        pieces = []
        for mesh, signature in sides:
            pieces.extend(self.extractFragments(mesh, signature))
        fragments = self.mergeEnclosedPieces(pieces, options.minFragmentFraction * labelled.GetNumberOfPoints())

        for fragment in fragments:
            for arrayName in ("ComponentId", "HostComponentId", "SheetDistance"):
                fragment.GetPointData().RemoveArray(arrayName)
        return fragments

    def mergePolyData(self, polyDatas: list[vtk.vtkPolyData]) -> vtk.vtkPolyData:
        """Join meshes into one (vtkAppendPolyData + vtkCleanPolyData).

        :param polyDatas: meshes to join.
        :return: joined mesh with coincident points merged.
        """
        append = vtk.vtkAppendPolyData()
        for polyData in polyDatas:
            append.AddInputData(polyData)
        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(append.GetOutputPort())
        clean.Update()
        result = vtk.vtkPolyData()
        result.DeepCopy(clean.GetOutput())
        return result

    #
    # Mesh helpers
    #

    @staticmethod
    def _ensureTriangles(polyData: vtk.vtkPolyData) -> vtk.vtkPolyData:
        """Return the mesh as triangles only (vertices, lines and strips are dropped/converted)."""
        polys = polyData.GetPolys()
        isTriangles = (polyData.GetNumberOfVerts() == 0 and polyData.GetNumberOfLines() == 0
                       and polyData.GetNumberOfStrips() == 0
                       and polys.GetConnectivityArray().GetNumberOfValues() == 3 * polys.GetNumberOfCells())
        result = vtk.vtkPolyData()
        if isTriangles:
            result.DeepCopy(polyData)
            return result
        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputData(polyData)
        triangleFilter.PassVertsOff()
        triangleFilter.PassLinesOff()
        triangleFilter.Update()
        result.DeepCopy(triangleFilter.GetOutput())
        return result

    def _connectedRegions(self, polyData: vtk.vtkPolyData
                          ) -> tuple[vtk.vtkPolyData, np.ndarray, np.ndarray, int]:
        """Label the connected regions of a triangle mesh (vtkPolyDataConnectivityFilter).

        :return: (triangle mesh copy without the RegionId array, point labels, cell labels,
            region count). Cell labels are taken from each triangle's first point, as the
            filter only writes point labels in this VTK version.
        """
        connectivity = vtk.vtkPolyDataConnectivityFilter()
        connectivity.SetInputData(polyData)
        connectivity.SetExtractionModeToAllRegions()
        connectivity.ColorRegionsOn()
        connectivity.Update()
        regions = self._ensureTriangles(connectivity.GetOutput())
        pointLabels = numpy_support.vtk_to_numpy(regions.GetPointData().GetArray("RegionId")).astype(np.int32)
        regions.GetPointData().RemoveArray("RegionId")
        regions.GetCellData().RemoveArray("RegionId")
        firstPoints = numpy_support.vtk_to_numpy(regions.GetPolys().GetConnectivityArray())[0::3]
        return regions, pointLabels, pointLabels[firstPoints], connectivity.GetNumberOfExtractedRegions()

    @staticmethod
    def _splitByCellLabel(polyData: vtk.vtkPolyData, cellLabels: np.ndarray) -> dict[int, vtk.vtkPolyData]:
        """Split a triangle mesh into one mesh per cell label, keeping all point data arrays.

        Done with numpy in one pass instead of one VTK filter run per label, which matters for
        segmentations with many small islands.
        """
        triangles = numpy_support.vtk_to_numpy(polyData.GetPolys().GetConnectivityArray()).reshape(-1, 3)
        allPoints = numpy_support.vtk_to_numpy(polyData.GetPoints().GetData())
        pointData = polyData.GetPointData()
        normalsArray = pointData.GetNormals()
        scalarsArray = pointData.GetScalars()
        dataArrays = [pointData.GetArray(i) for i in range(pointData.GetNumberOfArrays())
                      if pointData.GetArray(i) is not None]

        order = np.argsort(cellLabels, kind="stable")
        labels, starts = np.unique(cellLabels[order], return_index=True)
        ends = np.append(starts[1:], len(order))

        result = {}
        for label, start, end in zip(labels, starts, ends):
            labelTriangles = triangles[order[start:end]]
            usedPointIds, localTriangles = np.unique(labelTriangles, return_inverse=True)
            localTriangles = localTriangles.reshape(-1, 3)

            piece = vtk.vtkPolyData()
            points = vtk.vtkPoints()
            points.SetData(numpy_support.numpy_to_vtk(allPoints[usedPointIds], deep=True))
            piece.SetPoints(points)
            cells = vtk.vtkCellArray()
            offsets = np.arange(0, 3 * len(localTriangles) + 1, 3, dtype=np.int64)
            cells.SetData(numpy_support.numpy_to_vtk(offsets, deep=True, array_type=vtk.VTK_ID_TYPE),
                          numpy_support.numpy_to_vtk(localTriangles.ravel().astype(np.int64), deep=True,
                                                     array_type=vtk.VTK_ID_TYPE))
            piece.SetPolys(cells)

            for array in dataArrays:
                values = numpy_support.vtk_to_numpy(array)[usedPointIds]
                newArray = numpy_support.numpy_to_vtk(values, deep=True, array_type=array.GetDataType())
                newArray.SetName(array.GetName())
                piece.GetPointData().AddArray(newArray)
                if array is normalsArray:
                    piece.GetPointData().SetNormals(newArray)
                elif array is scalarsArray:
                    piece.GetPointData().SetScalars(newArray)
            result[int(label)] = piece
        return result

    @staticmethod
    def _addIntPointArray(polyData: vtk.vtkPolyData, name: str, values: np.ndarray) -> None:
        """Add (or replace) an int32 point data array."""
        array = numpy_support.numpy_to_vtk(np.ascontiguousarray(values, dtype=np.int32), deep=True,
                                           array_type=vtk.VTK_INT)
        array.SetName(name)
        polyData.GetPointData().AddArray(array)


#
# OsteotomyCutsTest
#


class OsteotomyCutsTest(ScriptedLoadableModuleTest):
    """Tests for the Osteotomy Cuts module. Synthetic geometry only, never patient data."""

    def setUp(self):
        """Reset the state by clearing the scene."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run every test_* method, each on a cleared scene."""
        testNames = sorted(name for name in dir(self) if name.startswith("test_"))
        for testName in testNames:
            self.setUp()
            getattr(self, testName)()
            print(f"PASSED {testName}")  # delayDisplay shows nothing when headless

    #
    # Helpers
    #

    @staticmethod
    def _points(polyData: vtk.vtkPolyData) -> np.ndarray:
        return numpy_support.vtk_to_numpy(polyData.GetPoints().GetData())

    @staticmethod
    def _cellNormals(polyData: vtk.vtkPolyData) -> np.ndarray:
        return numpy_support.vtk_to_numpy(polyData.GetCellData().GetNormals())

    def _assertNormalsConsistent(self, sheet: vtk.vtkPolyData) -> None:
        """Adjacent triangles must have normals on the same side (no flipped triangle)."""
        normals = self._cellNormals(sheet)
        sheet.BuildLinks()
        for cellId in range(sheet.GetNumberOfCells()):
            cellPoints = sheet.GetCell(cellId).GetPointIds()
            for k in range(3):
                edgeNeighbours = vtk.vtkIdList()
                sheet.GetCellEdgeNeighbors(cellId, cellPoints.GetId(k), cellPoints.GetId((k + 1) % 3),
                                           edgeNeighbours)
                for n in range(edgeNeighbours.GetNumberOfIds()):
                    # Neighbours across an edge may meet at a fold, but never at more than 90 degrees
                    # here; a flipped triangle would give a strongly negative dot product.
                    self.assertGreater(np.dot(normals[cellId], normals[edgeNeighbours.GetId(n)]), -0.5)

    #
    # Parameter node
    #

    def test_parameterNodeDefaults(self):
        """The parameter node starts with no view direction and stores a captured one."""
        logic = OsteotomyCutsLogic()
        parameterNode = logic.getParameterNode()

        self.assertFalse(isDirectionCaptured(parameterNode.viewDirection))
        self.assertEqual(parameterNode.directionMode, DirectionMode.VIEW)
        self.assertEqual(parameterNode.options.extension, 0.0)
        self.assertAlmostEqual(parameterNode.options.minFragmentFraction, 0.001)

        parameterNode.viewDirection = (0.0, 1.0, 0.0)
        parameterNode.directionMode = DirectionMode.LINE
        reread = logic.getParameterNode()
        self.assertTrue(isDirectionCaptured(reread.viewDirection))
        self.assertEqual(tuple(reread.viewDirection), (0.0, 1.0, 0.0))
        self.assertEqual(reread.directionMode, DirectionMode.LINE)

        parameterNode.viewDirection = NOT_CAPTURED
        self.assertFalse(isDirectionCaptured(logic.getParameterNode().viewDirection))

        self.delayDisplay("test_parameterNodeDefaults passed")

    #
    # Step 2: cutting sheet
    #

    def test_sheet_straight(self):
        """A two-point path gives one extended strip of 3 quads covering the extent."""
        logic = OsteotomyCutsLogic()
        extent = 50.0
        sheet = logic.buildSheetPolyData(np.array([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
                                         np.array([0.0, 0.0, 2.0]), extent)

        # 2 path points + 2 end extensions, each extruded to an A and a B vertex
        self.assertEqual(sheet.GetNumberOfPoints(), 8)
        self.assertEqual(sheet.GetNumberOfCells(), 6)
        bounds = sheet.GetBounds()
        np.testing.assert_allclose(bounds, (-60.0, 60.0, 0.0, 0.0, -50.0, 50.0), atol=1e-9)

        # The sheet is the plane y = 0: every normal is +/-y, and all on the same side
        normals = self._cellNormals(sheet)
        np.testing.assert_allclose(np.abs(normals[:, 1]), 1.0, atol=1e-9)
        self.assertTrue(np.all(normals[:, 1] * normals[0, 1] > 0))
        self.assertIsNotNone(sheet.GetPointData().GetNormals())
        self.delayDisplay("test_sheet_straight passed")

    def test_sheet_perPointDirections(self):
        """Per-point directions extrude each path point along its own (normalised) direction."""
        logic = OsteotomyCutsLogic()
        extent = 20.0
        path = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 5.0, 0.0]])
        directions = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 3.0]])
        sheet = logic.buildSheetPolyData(path, directions, extent)

        points = self._points(sheet)
        unitDirections = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
        for i in range(len(path)):
            # Index offset of 1 pair for the start extension
            np.testing.assert_allclose(points[2 * (i + 1)], path[i] - extent * unitDirections[i], atol=1e-9)
            np.testing.assert_allclose(points[2 * (i + 1) + 1], path[i] + extent * unitDirections[i], atol=1e-9)

        # End extensions continue the path outward, perpendicular to the end direction
        startPoint = (points[0] + points[1]) / 2.0
        np.testing.assert_allclose(startPoint, [-extent, 0.0, 0.0], atol=1e-9)
        endPoint = (points[-2] + points[-1]) / 2.0
        endTangent = (path[2] - path[1]) / np.linalg.norm(path[2] - path[1])
        np.testing.assert_allclose(endPoint, path[2] + extent * endTangent, atol=1e-9)
        self._assertNormalsConsistent(sheet)
        self.delayDisplay("test_sheet_perPointDirections passed")

    def test_sheet_closed(self):
        """A closed square path gives a tube of 4 quads, not extended, normals all outward or all inward."""
        logic = OsteotomyCutsLogic()
        square = np.array([[-5.0, -5.0, 0.0], [5.0, -5.0, 0.0], [5.0, 5.0, 0.0], [-5.0, 5.0, 0.0],
                           [-5.0, -5.0, 0.0]])  # repeated first point is dropped
        sheet = logic.buildSheetPolyData(square, np.array([0.0, 0.0, 1.0]), 30.0, closed=True)

        self.assertEqual(sheet.GetNumberOfPoints(), 8)
        self.assertEqual(sheet.GetNumberOfCells(), 8)
        np.testing.assert_allclose(sheet.GetBounds(), (-5.0, 5.0, -5.0, 5.0, -30.0, 30.0), atol=1e-9)

        centres = vtk.vtkCellCenters()
        centres.SetInputData(sheet)
        centres.Update()
        outward = self._points(centres.GetOutput()).copy()
        outward[:, 2] = 0.0
        sides = np.sign(np.einsum("ij,ij->i", self._cellNormals(sheet), outward))
        self.assertTrue(np.all(sides == sides[0]))
        self.delayDisplay("test_sheet_closed passed")

    def test_sheet_foldNormalsConsistent(self):
        """An L-shaped and a zig-zag path keep consistent normals across the folds."""
        logic = OsteotomyCutsLogic()
        lShape = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [20.0, 20.0, 0.0]])
        zigZag = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 0.0], [20.0, 0.0, 0.0], [30.0, 10.0, 0.0]])
        for path in (lShape, zigZag):
            sheet = logic.buildSheetPolyData(path, np.array([0.0, 0.0, 1.0]), 40.0)
            self._assertNormalsConsistent(sheet)
            # No point splitting: every A/B vertex is kept once
            self.assertEqual(sheet.GetNumberOfPoints(), 2 * (len(path) + 2))
        self.delayDisplay("test_sheet_foldNormalsConsistent passed")

    def test_sheet_errors(self):
        """Invalid inputs raise ValueError."""
        logic = OsteotomyCutsLogic()
        up = np.array([0.0, 0.0, 1.0])
        line = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        invalidCases = {
            "single point": (np.array([[0.0, 0.0, 0.0]]), up, 10.0, False),
            "duplicate points": (np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]), up, 10.0, False),
            "segment parallel to direction": (np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]), up, 10.0, False),
            "nearly parallel (2 degrees)": (np.array([[0.0, 0.0, 0.0], [np.sin(np.radians(2.0)), 0.0, np.cos(np.radians(2.0))]]),
                                            up, 10.0, False),
            "zero direction": (line, np.zeros(3), 10.0, False),
            "wrong direction count": (line, np.ones((3, 3)), 10.0, False),
            "zero extent": (line, up, 0.0, False),
            "closed with two points": (line, up, 10.0, True),
            "not (N, 3)": (np.zeros((2, 2)), up, 10.0, False),
        }
        for name, (path, directions, extent, closed) in invalidCases.items():
            with self.assertRaises(ValueError, msg=name):
                logic.buildSheetPolyData(path, directions, extent, closed)
        self.delayDisplay("test_sheet_errors passed")

    def test_autoExtent(self):
        """The automatic extent is the bounding-box diagonal plus margin."""
        logic = OsteotomyCutsLogic()
        box = vtk.vtkCubeSource()
        box.SetXLength(30.0)
        box.SetYLength(40.0)
        box.SetZLength(120.0)
        box.Update()
        self.assertAlmostEqual(logic.computeAutoExtent(box.GetOutput()), 130.0)
        self.assertAlmostEqual(logic.computeAutoExtent(box.GetOutput(), margin=5.0), 135.0)
        with self.assertRaises(ValueError):
            logic.computeAutoExtent(vtk.vtkPolyData())
        self.delayDisplay("test_autoExtent passed")

    #
    # Step 3: geometric cut
    #

    @staticmethod
    def _box(halfSize: float = 50.0, level: int = 20) -> vtk.vtkPolyData:
        """Closed, finely tessellated triangle box centred at the origin."""
        box = vtk.vtkTessellatedBoxSource()
        box.SetBounds(-halfSize, halfSize, -halfSize, halfSize, -halfSize, halfSize)
        box.SetLevel(level)
        box.DuplicateSharedPointsOff()
        box.QuadsOff()
        clean = vtk.vtkCleanPolyData()  # merge any remaining coincident points: one closed shell
        clean.SetInputConnection(box.GetOutputPort())
        clean.Update()
        return clean.GetOutput()

    @staticmethod
    def _sphere(radius: float, centre=(0.0, 0.0, 0.0), resolution: int = 64) -> vtk.vtkPolyData:
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(radius)
        sphere.SetCenter(*centre)
        sphere.SetThetaResolution(resolution)
        sphere.SetPhiResolution(resolution)
        sphere.Update()
        return sphere.GetOutput()

    @staticmethod
    def _append(*polyDatas: vtk.vtkPolyData) -> vtk.vtkPolyData:
        append = vtk.vtkAppendPolyData()
        for polyData in polyDatas:
            append.AddInputData(polyData)
        append.Update()
        return append.GetOutput()

    def _cut(self, polyData, paths, direction, closed=False, options=None):
        """Cut with one sheet per path, all extruded along the same direction."""
        logic = OsteotomyCutsLogic()
        extent = logic.computeAutoExtent(polyData)
        sheets = [logic.buildSheetPolyData(np.array(path, dtype=float), np.array(direction, dtype=float),
                                           extent, closed) for path in paths]
        return logic.cutPolyData(polyData, sheets, options or CutOptions())

    def _radii(self, polyData, centre=(0.0, 0.0, 0.0)) -> np.ndarray:
        return np.linalg.norm(self._points(polyData) - np.array(centre), axis=1)

    def test_cut_box_planar(self):
        """A straight path across the top of a box cuts it into two halves at the path."""
        box = self._box()
        pointsBefore = self._points(box).copy()
        x = 1.3  # off the grid lines, so no vertex lies exactly on the sheet
        fragments = self._cut(box, [[[x, -40.0, 50.0], [x, 40.0, 50.0]]], [0.0, 0.0, -1.0])

        self.assertEqual(len(fragments), 2)
        boundsList = sorted((fragment.GetBounds() for fragment in fragments), key=lambda b: b[0])
        self.assertAlmostEqual(boundsList[0][1], x, places=3)  # left piece ends at the cut
        self.assertAlmostEqual(boundsList[1][0], x, places=3)  # right piece starts at the cut
        for bounds in boundsList:  # both reach through the full height and depth
            np.testing.assert_allclose(bounds[2:], (-50.0, 50.0, -50.0, 50.0), atol=1e-6)
        # The input mesh is not modified
        np.testing.assert_array_equal(self._points(box), pointsBefore)
        self.assertIsNone(box.GetPointData().GetArray("SheetDistance"))

    def test_cut_box_Lshape(self):
        """An L-shaped path cuts the corner quadrant out of a box."""
        c = 20.3
        fragments = self._cut(self._box(), [[[c, -40.0, 50.0], [c, c, 50.0], [40.0, c, 50.0]]], [0.0, 0.0, -1.0])

        self.assertEqual(len(fragments), 2)
        corner = fragments[1]  # the smaller one
        bounds = corner.GetBounds()
        self.assertAlmostEqual(bounds[0], c, delta=0.5)
        self.assertAlmostEqual(bounds[1], 50.0, delta=1e-6)
        self.assertAlmostEqual(bounds[2], -50.0, delta=1e-6)
        self.assertAlmostEqual(bounds[3], c, delta=0.5)

    def test_cut_sphere_closedCurve(self):
        """A closed circle extruded through a sphere gives top cap, bottom cap and the band."""
        radius, circleRadius = 30.0, 10.0
        angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        z = np.sqrt(radius ** 2 - circleRadius ** 2)
        circle = np.column_stack([circleRadius * np.cos(angles), circleRadius * np.sin(angles),
                                  np.full_like(angles, z)])
        fragments = self._cut(self._sphere(radius), [circle], [0.0, 0.0, 1.0], closed=True)

        self.assertEqual(len(fragments), 3)
        band = fragments[0]
        caps = sorted(fragments[1:], key=lambda f: f.GetBounds()[4])
        self.assertLess(caps[0].GetBounds()[5], 0.0)  # bottom cap
        self.assertGreater(caps[1].GetBounds()[4], 0.0)  # top cap
        self.assertGreater(band.GetBounds()[1], circleRadius)

    def test_multipleSheets(self):
        """Two crossing sheets cut a box into four quadrants."""
        x, y = 1.3, 2.7
        fragments = self._cut(self._box(), [[[x, -40.0, 50.0], [x, 40.0, 50.0]],
                                            [[-40.0, y, 50.0], [40.0, y, 50.0]]], [0.0, 0.0, -1.0])

        self.assertEqual(len(fragments), 4)
        quadrants = set()
        for fragment in fragments:
            centre = self._points(fragment).mean(axis=0)
            quadrants.add((bool(centre[0] > x), bool(centre[1] > y)))
        self.assertEqual(len(quadrants), 4)

    def test_smallSpecksDropped(self):
        """A free-standing speck below the minimum fragment size is discarded."""
        speck = self._sphere(1.0, centre=(80.0, 0.0, 0.0), resolution=8)
        options = CutOptions()
        options.minFragmentFraction = 0.05
        fragments = self._cut(self._append(self._box(), speck), [[[1.3, -40.0, 50.0], [1.3, 40.0, 50.0]]],
                              [0.0, 0.0, -1.0], options=options)

        self.assertEqual(len(fragments), 2)
        for fragment in fragments:
            self.assertLess(fragment.GetBounds()[1], 51.0)

    def test_labelEnclosedComponents(self):
        """An inner sphere is labelled as enclosed by the outer one; an outside sphere is free."""
        logic = OsteotomyCutsLogic()
        mesh = self._append(self._sphere(30.0), self._sphere(10.0), self._sphere(5.0, centre=(60.0, 0.0, 0.0)))
        labelled = logic.labelEnclosedComponents(mesh, 0.001)

        components = numpy_support.vtk_to_numpy(labelled.GetPointData().GetArray("ComponentId"))
        hosts = numpy_support.vtk_to_numpy(labelled.GetPointData().GetArray("HostComponentId"))
        radii = self._radii(labelled)
        outer = components[np.argmin(np.abs(radii - 30.0))]
        inner = components[np.argmin(np.abs(radii - 10.0))]
        outside = components[np.argmax(radii)]
        self.assertEqual(len({int(outer), int(inner), int(outside)}), 3)
        self.assertTrue(np.all(hosts[components == inner] == outer))
        self.assertTrue(np.all(hosts[components == outer] == -1))
        self.assertTrue(np.all(hosts[components == outside] == -1))

    def test_enclosedShell(self):
        """Cutting a sphere with an inner shell gives 2 fragments, each with its half of the shell."""
        x = 1.3
        mesh = self._append(self._sphere(30.0), self._sphere(10.0))
        fragments = self._cut(mesh, [[[x, -20.0, 29.0], [x, 20.0, 29.0]]], [0.0, 0.0, -1.0])

        self.assertEqual(len(fragments), 2)
        for fragment in fragments:
            radii = self._radii(fragment)
            self.assertTrue(np.any(np.abs(radii - 30.0) < 0.5), "outer shell part missing")
            self.assertTrue(np.any(np.abs(radii - 10.0) < 0.5), "inner shell part missing")
            xs = self._points(fragment)[:, 0]
            self.assertTrue(np.all(xs <= x + 1e-3) or np.all(xs >= x - 1e-3), "fragment crosses the cut")

    def test_enclosedShell_uncut(self):
        """An inner shell entirely on one side of the cut is merged into that side's fragment."""
        x = 1.3
        innerCentre = (15.0, 0.0, 0.0)
        mesh = self._append(self._sphere(30.0), self._sphere(5.0, centre=innerCentre))
        fragments = self._cut(mesh, [[[x, -20.0, 29.0], [x, 20.0, 29.0]]], [0.0, 0.0, -1.0])

        self.assertEqual(len(fragments), 2)
        for fragment in fragments:
            hasInner = bool(np.any(np.abs(self._radii(fragment, innerCentre) - 5.0) < 0.1))
            onPositiveSide = bool(self._points(fragment)[:, 0].mean() > x)
            self.assertEqual(hasInner, onPositiveSide)

    def test_mergePolyData(self):
        """Joining two meshes keeps all their points and cells."""
        logic = OsteotomyCutsLogic()
        a, b = self._sphere(5.0), self._sphere(5.0, centre=(20.0, 0.0, 0.0))
        merged = logic.mergePolyData([a, b])
        self.assertEqual(merged.GetNumberOfPoints(), a.GetNumberOfPoints() + b.GetNumberOfPoints())
        self.assertEqual(merged.GetNumberOfCells(), a.GetNumberOfCells() + b.GetNumberOfCells())

    def test_cut_errors(self):
        """No sheet or an empty mesh raises ValueError."""
        logic = OsteotomyCutsLogic()
        with self.assertRaises(ValueError):
            logic.cutPolyData(self._box(), [], CutOptions())
        sheet = logic.buildSheetPolyData(np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]),
                                         np.array([0.0, 0.0, 1.0]), 10.0)
        with self.assertRaises(ValueError):
            logic.cutPolyData(vtk.vtkPolyData(), [sheet], CutOptions())
