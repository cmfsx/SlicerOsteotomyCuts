import enum
from dataclasses import dataclass
from typing import Annotated, Callable, Optional

import numpy as np
import qt
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

# Called with (percent 0-100, message) to report progress of long operations
ProgressCallback = Callable[[int, str], None]

# Node reference roles linking cut paths, their input model and their fragments. Node
# references are saved with the scene (subject hierarchy item IDs are not stable across reloads).
FRAGMENT_REFERENCE_ROLE = "OsteotomyCuts.Fragment"  # curve -> fragments (several)
INPUT_REFERENCE_ROLE = "OsteotomyCuts.Input"  # curve / fragment -> model that was cut
CURVE_REFERENCE_ROLE = "OsteotomyCuts.Curve"  # fragment -> curve that produced it

# Okabe-Ito colour-blind-safe palette followed by further distinct colours
FRAGMENT_COLOURS = (
    (0.90, 0.62, 0.00), (0.34, 0.71, 0.91), (0.00, 0.62, 0.45), (0.94, 0.89, 0.26),
    (0.00, 0.45, 0.70), (0.84, 0.37, 0.00), (0.80, 0.47, 0.65), (0.55, 0.34, 0.16),
    (0.60, 0.60, 0.60), (0.58, 0.40, 0.74), (0.74, 0.74, 0.13), (0.09, 0.75, 0.81),
)


def isDirectionCaptured(direction: tuple[float, float, float]) -> bool:
    """Return True if a stored direction holds a usable (non-zero) vector."""
    return any(abs(component) > 1e-9 for component in direction)


@parameterPack
class CutOptions:
    """Options for one cut. With kerfWidth = 0 and depth = 0 the cut is the Phase 1 zero-width
    through-cut."""

    # Distance (mm) the sheet extends past the path; 0 = automatic (bounding-box diagonal)
    extension: Annotated[float, WithinRange(0.0, 10000.0)] = 0.0
    # Free-standing pieces smaller than this fraction of the model's points are discarded
    minFragmentFraction: Annotated[float, WithinRange(0.0, 0.5)] = 0.001
    # Saw blade width (mm): bone closer than kerfWidth / 2 to the sheet is removed
    kerfWidth: Annotated[float, WithinRange(0.0, 10.0)] = 0.0
    # Cut depth (mm) from the path points along the extrusion direction; 0 = through the model
    depth: Annotated[float, WithinRange(0.0, 1000.0)] = 0.0
    # Close the cut faces so that fragments are watertight
    capCutFaces: bool = True
    # Maximum edge length (mm) near the sheet before cutting; 0 = automatic (kerfWidth / 2)
    refineEdgeLength: Annotated[float, WithinRange(0.0, 10.0)] = 0.0


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

    # Delay after the last point edit before the sheet preview is rebuilt
    PREVIEW_DELAY_MS = 80

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialised."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self._observedMarkupsNodes = []  # curve and line whose point edits refresh the GUI
        self._resultMessage = ""  # outcome of the last action, shown while inputs are valid
        self._previewTimer = None  # throttles sheet preview rebuilds while points are dragged
        self._snapping = False  # guards against reacting to our own snapping edits
        self._reconnectAfterImport = False  # GUI was connected when a scene import started

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
        # A loaded scene may bring a parameter node saved by an earlier version of the module
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartImportEvent, self.onSceneStartImport)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndImportEvent, self.onSceneEndImport)

        # The direction mode enum has no ready-made radio button connector, so it is wired here
        self.ui.directionViewRadioButton.connect("toggled(bool)", self.onDirectionModeToggled)
        self.ui.directionLineRadioButton.connect("toggled(bool)", self.onDirectionModeToggled)

        self.ui.cutCurveSelector.connect("nodeAddedByUser(vtkMRMLNode*)", self.onCutCurveAdded)
        self.ui.captureViewDirectionButton.connect("clicked(bool)", self.onCaptureViewDirection)
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.undoButton.connect("clicked(bool)", self.onUndoButton)
        self.ui.mergeButton.connect("clicked(bool)", self.onMergeButton)
        self.ui.mergeFragmentsSelector.connect("checkedNodesChanged()", self._updateMergeButton)
        self.ui.sheetOpacitySliderWidget.connect("valueChanged(double)", self._applySheetOpacity)

        self._previewTimer = qt.QTimer()
        self._previewTimer.setSingleShot(True)
        self._previewTimer.setInterval(self.PREVIEW_DELAY_MS)
        self._previewTimer.connect("timeout()", self._updatePreview)

        # Make sure parameter node is initialised (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        if self._previewTimer:
            self._previewTimer.stop()
        self.removeObservers()
        self._observedMarkupsNodes = []

    def enter(self) -> None:
        """Called each time the user opens this module."""
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode.parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateGuiFromParameterNode)
        self._observeMarkupsNodes([])
        if self._previewTimer:
            self._previewTimer.stop()

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        if self.parent.isEntered:
            self.initializeParameterNode()

    def onSceneStartImport(self, caller, event) -> None:
        """Called before a scene is loaded: let go of the parameter node while it may be replaced
        by one saved with fewer parameters (reading it half-loaded would fail)."""
        self._reconnectAfterImport = self._parameterNodeGuiTag is not None
        self.setParameterNode(None)

    def onSceneEndImport(self, caller, event) -> None:
        """Called after a scene is loaded: upgrade the parameter node and reconnect the GUI."""
        if self._reconnectAfterImport:
            self.initializeParameterNode()  # logic.getParameterNode() adds missing parameters
        else:
            self.logic.getParameterNode()  # upgrade now; the GUI connects on the next enter()
        self._reconnectAfterImport = False

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode: Optional[OsteotomyCutsParameterNode]) -> None:
        """Set and observe parameter node, so that the GUI follows parameter changes."""
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            # The raw MRML node is observed, not the wrapper: VTKObservationMixin keeps a
            # reference to every object it has observed, which would keep old wrappers alive.
            self.removeObserver(self._parameterNode.parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateGuiFromParameterNode)
        self._observeMarkupsNodes([])
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Widgets with a "SlicerParameterName" property in the .ui file are connected here
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode.parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateGuiFromParameterNode)
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

        self._observeMarkupsNodes([self._parameterNode.cutCurve, self._parameterNode.directionLine])
        self._updateActionState()
        self._schedulePreview()

    def _onMarkupsModified(self, caller=None, event=None) -> None:
        """Points of the cut path or direction line changed."""
        self._updateActionState()
        self._schedulePreview()

    def _onCurvePointsPlaced(self, caller=None, event=None) -> None:
        """After a point is placed or dragged, put the cut path back on the model surface."""
        parameterNode = self._parameterNode
        if (self._snapping or not parameterNode or not parameterNode.snapToSurface
                or parameterNode.inputModel is None or parameterNode.cutCurve is None
                or caller is None or caller.GetID() != parameterNode.cutCurve.GetID()):
            return
        self._snapping = True
        try:
            self.logic.snapCurveToSurface(parameterNode.cutCurve, parameterNode.inputModel)
        except ValueError:
            pass  # empty model or non-linear transform: validateInputs already reports it
        finally:
            self._snapping = False

    def _schedulePreview(self) -> None:
        """Rebuild the sheet preview shortly, once point edits pause."""
        if self._previewTimer:
            self._previewTimer.start()

    def _updatePreview(self) -> None:
        """Show, update or hide the cutting sheet preview."""
        if not self._parameterNode:
            return
        if self.logic.updateSheetModel(self._parameterNode) is not None:
            self._applySheetOpacity()

    def _applySheetOpacity(self, value: Optional[float] = None) -> None:
        """Set the opacity of the sheet preview from the slider (display only, not a parameter)."""
        sheetNode = self._parameterNode.sheetModel if self._parameterNode else None
        if sheetNode is not None and sheetNode.GetDisplayNode() is not None:
            sheetNode.GetDisplayNode().SetOpacity(self.ui.sheetOpacitySliderWidget.value)

    def _updateActionState(self, caller=None, event=None) -> None:
        """Enable Apply / Undo and show why a cut cannot run yet."""
        if not self._parameterNode:
            return
        reason = self.logic.validateInputs(self._parameterNode)
        self.ui.applyButton.enabled = reason is None
        self.ui.applyButton.toolTip = reason or _("Cut the model along the cutting sheet.")
        self.ui.undoButton.enabled = bool(self.logic.getCurveResult(self._parameterNode.cutCurve))
        self.ui.statusLabel.text = reason or self._resultMessage or _("Ready to cut.")
        self._updateMergeButton()

    def _updateMergeButton(self) -> None:
        self.ui.mergeButton.enabled = len(self.ui.mergeFragmentsSelector.checkedNodes()) >= 2

    def _observeMarkupsNodes(self, nodes: list) -> None:
        """Refresh the GUI and preview when points of these markups nodes change, and snap
        cut path points to the surface when a placement or drag ends."""
        nodes = [node for node in nodes if node is not None]
        if [n.GetID() for n in nodes] == [n.GetID() for n in self._observedMarkupsNodes]:
            return
        markups = slicer.vtkMRMLMarkupsNode
        observations = [(event, self._onMarkupsModified) for event in
                        (markups.PointAddedEvent, markups.PointRemovedEvent, markups.PointModifiedEvent)]
        observations += [(event, self._onCurvePointsPlaced) for event in
                         (markups.PointEndInteractionEvent, markups.PointPositionDefinedEvent)]
        for node in self._observedMarkupsNodes:
            for event, callback in observations:
                self.removeObserver(node, event, callback)
        self._observedMarkupsNodes = nodes
        for node in nodes:
            for event, callback in observations:
                self.addObserver(node, event, callback)

    def onCutCurveAdded(self, curveNode) -> None:
        """New cut paths are polylines: straight segments between the placed points."""
        curveNode.SetCurveTypeToLinear()

    def _requireParameterNode(self) -> OsteotomyCutsParameterNode:
        """Return the parameter node, reconnecting it if a scene close or load left none."""
        if self._parameterNode is None:
            self.initializeParameterNode()
        return self._parameterNode

    def onCaptureViewDirection(self) -> None:
        """Store the viewing direction of the first 3D view."""
        with slicer.util.tryWithErrorDisplay(_("Failed to capture the view direction."), waitCursor=True):
            viewNode = slicer.app.layoutManager().threeDWidget(0).mrmlViewNode()
            self.logic.captureViewDirection(self._requireParameterNode(), viewNode)

    def onApplyButton(self) -> None:
        """Cut the model, with a progress dialog."""
        progress = slicer.util.createProgressDialog(labelText=_("Cutting..."), maximum=100)

        def reportProgress(percent: int, message: str) -> None:
            progress.labelText = message
            progress.value = percent
            slicer.app.processEvents()

        try:
            with slicer.util.tryWithErrorDisplay(_("Failed to cut the model."), waitCursor=True):
                fragments = self.logic.applyCut(self._requireParameterNode(), reportProgress)
                self._resultMessage = _("{count} fragments created.").format(count=len(fragments))
        finally:
            progress.close()
            self._updateActionState()

    def onUndoButton(self) -> None:
        """Remove the fragments of the selected cut path and show its input model again."""
        with slicer.util.tryWithErrorDisplay(_("Failed to undo the cut."), waitCursor=True):
            self.logic.removeCutResult(self._requireParameterNode().cutCurve)
            self._resultMessage = _("Cut undone.")
        self._updateActionState()

    def onMergeButton(self) -> None:
        """Join the ticked fragments into one model."""
        with slicer.util.tryWithErrorDisplay(_("Failed to merge the fragments."), waitCursor=True):
            self._requireParameterNode()
            merged =self.logic.mergeFragments(list(self.ui.mergeFragmentsSelector.checkedNodes()))
            self._resultMessage = _("Fragments merged into {name}.").format(name=merged.GetName())
        self._updateActionState()


#
# OsteotomyCutsLogic
#


class OsteotomyCutsLogic(ScriptedLoadableModuleLogic):
    """Computation for the Osteotomy Cuts module. Runs without the GUI (headless)."""

    def __init__(self) -> None:
        """Initialise the logic."""
        ScriptedLoadableModuleLogic.__init__(self)
        # (cache key, vtkCellLocator) for snapping to the last used model surface
        self._surfaceLocatorCache = None

    # A path segment closer than this angle to its extrusion direction is rejected, because the
    # sheet would degenerate to a sliver there.
    MIN_SEGMENT_DIRECTION_ANGLE_DEG = 5.0

    # Consecutive path points closer than this (mm) are treated as duplicates
    DUPLICATE_POINT_TOLERANCE = 1e-6

    # Points closer than this (mm) to the surface are not moved when snapping
    SNAP_TOLERANCE = 1e-3

    # Limits for mesh refinement near the sheet (each iteration halves the edges it splits)
    MAX_REFINE_ITERATIONS = 30
    MAX_REFINED_POINTS = 20_000_000

    def getParameterNode(self) -> OsteotomyCutsParameterNode:
        """Return the module's parameter node, creating it if needed.

        Parameters added in a later version are filled in with their defaults first, so that
        scenes saved by an earlier version still load.
        """
        parameterNode = super().getParameterNode()
        self.addMissingParameters(parameterNode)
        return OsteotomyCutsParameterNode(parameterNode)

    @staticmethod
    def addMissingParameters(parameterNode) -> list[str]:
        """Write default values for parameters the node does not have yet (scene upgrade).

        parameterNodeWrapper only writes defaults when a whole parameter (e.g. the ``options``
        pack) is missing; a pack saved with fewer fields fails to read. Existing values are
        never changed.

        :param parameterNode: the raw vtkMRMLScriptedModuleNode.
        :return: names of the parameters that were added.
        """
        defaults = slicer.vtkMRMLScriptedModuleNode()
        OsteotomyCutsParameterNode(defaults)  # writes every default value into the temporary node
        existing = set(parameterNode.GetParameterNames())
        if not existing:
            return []  # new node: the wrapper writes all defaults itself
        added = [name for name in defaults.GetParameterNames() if name not in existing]
        for name in added:
            parameterNode.SetParameter(name, defaults.GetParameter(name))
        return added

    #
    # Geometry (no MRML)
    #

    def buildSheetPolyData(self, pathPoints: np.ndarray, directions: np.ndarray,
                           extent: float, closed: bool = False,
                           depth: Optional[float | np.ndarray] = None) -> vtk.vtkPolyData:
        """Build a ruled cutting sheet by extruding a polyline.

        The direction d points into the model (away from the viewer). Each path point P is
        extruded outward to A = P - extent * d and inward to B = P + depth * d (``depth``
        defaults to ``extent``, a through-cut). An open path is also extended by ``extent`` at
        both ends, along the end tangent with its component along d removed, so that the sheet
        edges lie outside the model.

        Output point order is A, B pairs along the (extended) path:
        ``[A_start, B_start, A_0, B_0, ..., A_last, B_last, A_end, B_end]`` for an open path and
        ``[A_0, B_0, ..., A_last, B_last]`` for a closed path.

        :param pathPoints: (N, 3) world points along the cut path, N >= 2 (N >= 3 if closed).
            Consecutive duplicates are removed.
        :param directions: (3,) one extrusion direction, or (N, 3) one per path point
            (Phase 3 BSSO templates use a different direction per segment). Normalised here.
        :param extent: distance (mm) the sheet reaches along +/- direction and past the ends.
        :param closed: join the last point to the first and do not extend the ends.
        :param depth: inward reach (mm) along d from the path: one value, or (N,) one per path
            point (for templates). None means ``extent`` (through-cut).
        :return: triangulated sheet with consistent winding, point and cell normals.
        :raises ValueError: too few distinct points, a zero or malformed direction, a
            non-positive extent or depth, or a path segment (nearly) parallel to its direction.
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

        depths = np.full(len(points), float(extent)) if depth is None else np.asarray(depth, dtype=float)
        if depths.ndim == 0:
            depths = np.full(len(points), float(depths))
        elif depths.shape != (len(points),):
            raise ValueError(_("Depth must be a single value or one value per path point."))
        if np.any(~(depths > 0)):
            raise ValueError(_("The cut depth must be positive."))

        # Remove consecutive duplicates (and, for a closed path, a repeated first point at the end)
        if len(points) > 0:
            keep = np.ones(len(points), dtype=bool)
            keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > self.DUPLICATE_POINT_TOLERANCE
            points, dirs, depths = points[keep], dirs[keep], depths[keep]
        if closed and len(points) > 1 and np.linalg.norm(points[-1] - points[0]) <= self.DUPLICATE_POINT_TOLERANCE:
            points, dirs, depths = points[:-1], dirs[:-1], depths[:-1]

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
            depths = np.concatenate([depths[:1], depths, depths[-1:]])

        # Interleaved A_i (outward), B_i (inward) vertices
        vertices = np.empty((2 * len(points), 3))
        vertices[0::2] = points - extent * dirs
        vertices[1::2] = points + depths[:, np.newaxis] * dirs

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
        :param options: cut options; kerf removal (kerfWidth > 0) is added in Phase 2 step 3.
        :param arrayName: name of the distance array.
        :return: (positive side, negative side), both triangulated; either may be empty.
        :raises ValueError: for kerfWidth > 0, until kerf removal is implemented.
        """
        if options.kerfWidth > 0:
            raise ValueError(_("Cutting with a kerf width is not available yet. Set the kerf width to 0."))
        polyWithDistance.GetPointData().SetActiveScalars(arrayName)
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(polyWithDistance)
        clipper.SetValue(0.0)
        clipper.GenerateClippedOutputOn()
        clipper.Update()
        return (self._ensureTriangles(clipper.GetOutput()),
                self._ensureTriangles(clipper.GetClippedOutput()))

    @staticmethod
    def effectiveRefineEdgeLength(options: CutOptions) -> float:
        """Maximum edge length near the sheet: options.refineEdgeLength, or kerfWidth / 2 when
        that is 0 (automatic). 0 means no refinement (zero kerf, no explicit length)."""
        if options.refineEdgeLength > 0:
            return float(options.refineEdgeLength)
        return options.kerfWidth / 2.0

    def refineNearSheet(self, polyWithDistance: vtk.vtkPolyData, sheetPolyData: vtk.vtkPolyData,
                        maxEdgeLength: float, bandWidth: float,
                        arrayName: str = "SheetDistance") -> vtk.vtkPolyData:
        """Subdivide mesh edges near the sheet until they are no longer than maxEdgeLength.

        An edge is split at its midpoint when it is too long and may reach within bandWidth of
        the sheet (the distance is 1-Lipschitz, so the smallest distance along an edge of length
        L with end distances d0, d1 is at least (|d0| + |d1| - L) / 2). The split decision is
        made per edge, so both triangles sharing an edge agree: the result has no cracks and a
        closed mesh stays closed. Each triangle is re-split by how many of its edges were split
        (1, 2 or 3), keeping its orientation. Geometry is unchanged; the original points keep
        their ids. Point data is interpolated at new points (integer arrays take the value of
        the first edge end, as a whole edge lies in one component); the distance array is
        evaluated exactly at new points.

        :param polyWithDistance: mesh with the signed distance point array (computeSheetDistance).
        :param sheetPolyData: the cutting sheet, to evaluate the distance at new points.
        :param maxEdgeLength: target maximum edge length (mm) near the sheet.
        :param bandWidth: distance (mm) from the sheet within which edges are refined.
        :param arrayName: name of the distance array.
        :return: refined triangle mesh with all point data arrays, including exact distances.
        :raises ValueError: for a non-positive edge length, or if refinement would exceed
            MAX_REFINED_POINTS (edge length too small for the model).
        """
        if not maxEdgeLength > 0:
            raise ValueError(_("The maximum edge length must be positive."))
        mesh = self._ensureTriangles(polyWithDistance)
        points = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData()).astype(float)
        triangles = numpy_support.vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3).astype(np.int64)
        pointData = mesh.GetPointData()
        arrays = {}
        for i in range(pointData.GetNumberOfArrays()):
            array = pointData.GetArray(i)
            if array is not None:
                arrays[array.GetName()] = (numpy_support.vtk_to_numpy(array).copy(), array.GetDataType())
        if arrayName not in arrays:
            raise ValueError(_("The mesh has no sheet distance array."))
        normalsName = pointData.GetNormals().GetName() if pointData.GetNormals() else None
        scalarsName = pointData.GetScalars().GetName() if pointData.GetScalars() else None

        implicitDistance = vtk.vtkImplicitPolyDataDistance()
        implicitDistance.SetInput(sheetPolyData)

        for _iteration in range(self.MAX_REFINE_ITERATIONS):
            pointCount = len(points)
            distances = np.abs(arrays[arrayName][0])

            # Unique edges, and for each triangle the index of its edges (v0v1, v1v2, v2v0)
            triangleEdges = np.stack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=1)
            sortedEdges = np.sort(triangleEdges.reshape(-1, 2), axis=1)
            keys, edgeOfTriangle = np.unique(sortedEdges[:, 0] * pointCount + sortedEdges[:, 1], return_inverse=True)
            edgeStart, edgeEnd = keys // pointCount, keys % pointCount
            edgeOfTriangle = edgeOfTriangle.reshape(-1, 3)

            lengths = np.linalg.norm(points[edgeStart] - points[edgeEnd], axis=1)
            closestAlongEdge = (distances[edgeStart] + distances[edgeEnd] - lengths) / 2.0
            split = (lengths > maxEdgeLength) & (closestAlongEdge < bandWidth)
            splitCount = int(np.count_nonzero(split))
            if splitCount == 0:
                break
            if pointCount + splitCount > self.MAX_REFINED_POINTS:
                raise ValueError(_("Refining the mesh near the cut would create too many points. "
                                   "Increase the maximum edge length near the cut or the kerf width."))

            # New midpoints, with exact distances and interpolated point data
            splitStart, splitEnd = edgeStart[split], edgeEnd[split]
            midpoints = (points[splitStart] + points[splitEnd]) / 2.0
            for name, (values, dataType) in arrays.items():
                if name == arrayName:
                    newDistances = vtk.vtkDoubleArray()
                    implicitDistance.FunctionValue(numpy_support.numpy_to_vtk(midpoints, deep=True), newDistances)
                    newValues = numpy_support.vtk_to_numpy(newDistances).astype(values.dtype)
                elif np.issubdtype(values.dtype, np.floating):
                    newValues = (values[splitStart] + values[splitEnd]) / 2.0
                    if name == normalsName:
                        norms = np.linalg.norm(newValues, axis=1, keepdims=True)
                        newValues = newValues / np.where(norms > 0, norms, 1.0)
                else:
                    newValues = values[splitStart]
                arrays[name] = (np.concatenate([values, newValues.astype(values.dtype)]), dataType)
            points = np.vstack([points, midpoints])

            midpointOfEdge = np.full(len(keys), -1, dtype=np.int64)
            midpointOfEdge[split] = pointCount + np.arange(splitCount)
            triangles = self._splitTriangles(triangles, midpointOfEdge[edgeOfTriangle])

        return self._buildTriangleMesh(points, triangles, arrays, normalsName, scalarsName)

    @staticmethod
    def _splitTriangles(triangles: np.ndarray, midpoints: np.ndarray) -> np.ndarray:
        """Re-split triangles whose edges got midpoints, keeping the vertex order (orientation).

        :param triangles: (M, 3) vertex ids (v0, v1, v2).
        :param midpoints: (M, 3) midpoint id of edges (v0v1, v1v2, v2v0), or -1 if not split.
        :return: (M', 3) triangles.
        """
        isSplit = midpoints >= 0
        splitCounts = isSplit.sum(axis=1)
        result = [triangles[splitCounts == 0]]

        def rotated(rows: np.ndarray, shift: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            order = (shift[:, np.newaxis] + np.arange(3)) % 3
            return (np.take_along_axis(triangles[rows], order, axis=1),
                    np.take_along_axis(midpoints[rows], order, axis=1))

        # One split edge: rotate it to v0v1, then two triangles meeting at the midpoint
        rows = np.flatnonzero(splitCounts == 1)
        v, m = rotated(rows, np.argmax(isSplit[rows], axis=1))
        result += [np.column_stack([v[:, 0], m[:, 0], v[:, 2]]), np.column_stack([m[:, 0], v[:, 1], v[:, 2]])]

        # Two split edges: rotate them to v0v1 and v1v2, then the corner and the remaining quad
        rows = np.flatnonzero(splitCounts == 2)
        v, m = rotated(rows, (np.argmin(isSplit[rows], axis=1) + 1) % 3)
        result += [np.column_stack([m[:, 0], v[:, 1], m[:, 1]]), np.column_stack([v[:, 0], m[:, 0], m[:, 1]]),
                   np.column_stack([v[:, 0], m[:, 1], v[:, 2]])]

        # Three split edges: four triangles
        rows = np.flatnonzero(splitCounts == 3)
        v, m = triangles[rows], midpoints[rows]
        result += [np.column_stack([v[:, 0], m[:, 0], m[:, 2]]), np.column_stack([m[:, 0], v[:, 1], m[:, 1]]),
                   np.column_stack([m[:, 2], m[:, 1], v[:, 2]]), np.column_stack([m[:, 0], m[:, 1], m[:, 2]])]
        return np.vstack(result)

    @staticmethod
    def _buildTriangleMesh(points: np.ndarray, triangles: np.ndarray, arrays: dict,
                           normalsName: Optional[str], scalarsName: Optional[str]) -> vtk.vtkPolyData:
        """Assemble a vtkPolyData from numpy points, triangles and named point data arrays."""
        mesh = vtk.vtkPolyData()
        vtkPoints = vtk.vtkPoints()
        vtkPoints.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(points), deep=True))
        mesh.SetPoints(vtkPoints)
        cells = vtk.vtkCellArray()
        offsets = np.arange(0, 3 * len(triangles) + 1, 3, dtype=np.int64)
        cells.SetData(numpy_support.numpy_to_vtk(offsets, deep=True, array_type=vtk.VTK_ID_TYPE),
                      numpy_support.numpy_to_vtk(np.ascontiguousarray(triangles.ravel(), dtype=np.int64),
                                                 deep=True, array_type=vtk.VTK_ID_TYPE))
        mesh.SetPolys(cells)
        for name, (values, dataType) in arrays.items():
            array = numpy_support.numpy_to_vtk(np.ascontiguousarray(values), deep=True, array_type=dataType)
            array.SetName(name)
            mesh.GetPointData().AddArray(array)
            if name == normalsName:
                mesh.GetPointData().SetNormals(array)
            elif name == scalarsName:
                mesh.GetPointData().SetScalars(array)
        return mesh

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
                    options: CutOptions,
                    progressCallback: Optional[ProgressCallback] = None) -> list[vtk.vtkPolyData]:
        """Cut a mesh with one or more sheets into separate fragments (headless core).

        Each sheet is applied in turn to every current piece, then the pieces are split into
        connected fragments and enclosed pieces are merged into their host. Several sheets let
        Phase 3 cut several osteotomies at once. The input mesh is not modified.

        :param polyData: model mesh in world coordinates.
        :param sheets: cutting sheets from buildSheetPolyData.
        :param options: cut options.
        :param progressCallback: called with (percent, message) between stages.
        :return: fragment meshes, largest first.
        :raises ValueError: if there is no sheet or the mesh has no polygons.
        """
        report = progressCallback or (lambda percent, message: None)
        if not sheets:
            raise ValueError(_("At least one cutting sheet is needed."))
        triangles = self._ensureTriangles(polyData)
        if triangles.GetNumberOfCells() == 0:
            raise ValueError(_("The model has no surface polygons to cut."))

        report(5, _("Finding internal shells..."))
        labelled = self.labelEnclosedComponents(triangles, options.minFragmentFraction)
        sides = [(labelled, ())]
        for sheetIndex, sheet in enumerate(sheets):
            report(15 + int(60 * sheetIndex / len(sheets)), _("Cutting..."))
            nextSides = []
            for mesh, signature in sides:
                withDistance = self.computeSheetDistance(mesh, sheet)
                maxEdgeLength = self.effectiveRefineEdgeLength(options)
                if maxEdgeLength > 0:
                    withDistance = self.refineNearSheet(withDistance, sheet, maxEdgeLength,
                                                        options.kerfWidth / 2.0 + maxEdgeLength)
                positive, negative = self.splitByDistance(withDistance, options)
                for part, side in ((positive, 1), (negative, -1)):
                    if part.GetNumberOfCells() > 0:
                        nextSides.append((part, signature + (side,)))
            sides = nextSides

        report(75, _("Separating fragments..."))
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
    # Inputs and directions (MRML)
    #

    def getPathPoints(self, curveNode: vtkMRMLMarkupsCurveNode) -> np.ndarray:
        """Return the curve's points in world coordinates, (N, 3).

        For a linear curve these are the control points; for a spline, the sampled curve.

        :raises ValueError: if the curve has fewer than 2 control points.
        """
        if curveNode is None or curveNode.GetNumberOfControlPoints() < 2:
            raise ValueError(_("The cut path needs at least 2 points."))
        if curveNode.GetCurveType() == slicer.vtkCurveGenerator.CURVE_TYPE_LINEAR_SPLINE:
            # The sampled curve adds redundant points along straight segments
            return np.array(slicer.util.arrayFromMarkupsControlPoints(curveNode, world=True), dtype=float)
        return np.array(slicer.util.arrayFromMarkupsCurvePoints(curveNode, world=True), dtype=float)

    @staticmethod
    def isClosedCurve(curveNode: vtkMRMLMarkupsCurveNode) -> bool:
        """Return True for a closed curve (vtkMRMLMarkupsClosedCurveNode)."""
        return curveNode.IsA("vtkMRMLMarkupsClosedCurveNode")

    def directionFromLine(self, lineNode: vtkMRMLMarkupsLineNode) -> np.ndarray:
        """Return the unit vector from the line's first to its second point (world coordinates).

        :raises ValueError: if the line has fewer than 2 points or zero length.
        """
        if lineNode is None or lineNode.GetNumberOfControlPoints() < 2:
            raise ValueError(_("The direction line needs 2 points."))
        points = np.array(slicer.util.arrayFromMarkupsControlPoints(lineNode, world=True), dtype=float)
        return self._normalised(points[1] - points[0], _("The direction line has zero length."))

    def directionFromView(self, viewNode) -> np.ndarray:
        """Return the viewing direction (direction of projection) of a 3D view's camera.

        Uses only MRML nodes, so it also works without a GUI.

        :param viewNode: vtkMRMLViewNode of the 3D view.
        :raises ValueError: if the view has no camera.
        """
        cameraNode = slicer.modules.cameras.logic().GetViewActiveCameraNode(viewNode) if viewNode else None
        if cameraNode is None:
            raise ValueError(_("The 3D view has no camera."))
        return self._normalised(np.array(cameraNode.GetFocalPoint()) - np.array(cameraNode.GetPosition()),
                                _("The 3D view camera has no direction."))

    def captureViewDirection(self, parameterNode: OsteotomyCutsParameterNode, viewNode) -> None:
        """Store the current viewing direction of a 3D view as the extrusion direction."""
        parameterNode.viewDirection = tuple(float(c) for c in self.directionFromView(viewNode))

    def resolveDirection(self, parameterNode: OsteotomyCutsParameterNode) -> np.ndarray:
        """Return the extrusion direction for the current direction mode.

        :raises ValueError: if the view direction is not captured or the line is invalid.
        """
        if parameterNode.directionMode == DirectionMode.LINE:
            if parameterNode.directionLine is None:
                raise ValueError(_("Select a direction line."))
            return self.directionFromLine(parameterNode.directionLine)
        if not isDirectionCaptured(parameterNode.viewDirection):
            raise ValueError(_("Capture a view direction first."))
        return self._normalised(np.array(parameterNode.viewDirection), _("Capture a view direction first."))

    def validateInputs(self, parameterNode: OsteotomyCutsParameterNode) -> Optional[str]:
        """Check whether a cut can run.

        :return: None if it can, otherwise a message for the user saying what is missing.
        """
        inputModel = parameterNode.inputModel
        if inputModel is None:
            return _("Select a model to cut.")
        if inputModel.GetPolyData() is None or inputModel.GetPolyData().GetNumberOfPoints() == 0:
            return _("The selected model is empty.")
        transformNode = inputModel.GetParentTransformNode()
        if transformNode is not None and not transformNode.IsTransformToWorldLinear():
            return _("The model is under a non-linear transform. Harden the transform first.")
        curveNode = parameterNode.cutCurve
        if curveNode is None:
            return _("Select or create a cut path.")
        minPoints = 3 if self.isClosedCurve(curveNode) else 2
        if curveNode.GetNumberOfControlPoints() < minPoints:
            return _("Place at least {count} points on the cut path.").format(count=minPoints)
        if inputModel.GetNodeReferenceID(CURVE_REFERENCE_ROLE) == curveNode.GetID():
            return _("The model to cut was produced by this cut path. Select another model or path.")
        try:
            self.resolveDirection(parameterNode)
        except ValueError as error:
            return str(error)
        if parameterNode.options.depth > 0 and not parameterNode.options.kerfWidth > 0:
            # A zero-width cut that stops inside the model removes nothing and separates nothing
            return _("A limited cut depth needs a kerf width greater than 0.")
        return None

    def getWorldPolyData(self, modelNode: vtkMRMLModelNode) -> vtk.vtkPolyData:
        """Return a copy of the model mesh in world coordinates (the node is not modified).

        :raises ValueError: if the model is empty or under a non-linear transform.
        """
        polyData = modelNode.GetPolyData()
        if polyData is None or polyData.GetNumberOfPoints() == 0:
            raise ValueError(_("The selected model is empty."))
        result = vtk.vtkPolyData()
        transformNode = modelNode.GetParentTransformNode()
        if transformNode is None:
            result.DeepCopy(polyData)
            return result
        if not transformNode.IsTransformToWorldLinear():
            raise ValueError(_("The model is under a non-linear transform. Harden the transform first."))
        matrix = vtk.vtkMatrix4x4()
        transformNode.GetMatrixTransformToWorld(matrix)
        transform = vtk.vtkTransform()
        transform.SetMatrix(matrix)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputData(polyData)
        transformFilter.SetTransform(transform)
        transformFilter.Update()
        result.DeepCopy(transformFilter.GetOutput())
        return result

    def snapCurveToSurface(self, curveNode: vtkMRMLMarkupsCurveNode, modelNode: vtkMRMLModelNode) -> int:
        """Move each control point of the curve to the closest point on the model surface.

        Used after points are placed or dragged (also in slice views, where they are not on
        the surface), and by Phase 3 templates placing landmarks.

        :return: number of points that were moved.
        :raises ValueError: if the model is empty or under a non-linear transform.
        """
        locator = self._getSurfaceLocator(modelNode)
        position, closest = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        cellId, subId, distance2 = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
        moved = 0
        wasModifying = curveNode.StartModify()
        try:
            for i in range(curveNode.GetNumberOfControlPoints()):
                curveNode.GetNthControlPointPositionWorld(i, position)
                locator.FindClosestPoint(position, closest, cellId, subId, distance2)
                if float(distance2) > self.SNAP_TOLERANCE ** 2:
                    curveNode.SetNthControlPointPositionWorld(i, *closest)
                    moved += 1
        finally:
            curveNode.EndModify(wasModifying)
        return moved

    def _getSurfaceLocator(self, modelNode: vtkMRMLModelNode) -> vtk.vtkCellLocator:
        """Cell locator on the model surface in world coordinates, cached until the model changes."""
        transformNode = modelNode.GetParentTransformNode()
        polyData = modelNode.GetPolyData()
        matrixKey = None
        if transformNode is not None and transformNode.IsTransformToWorldLinear():
            matrix = vtk.vtkMatrix4x4()
            transformNode.GetMatrixTransformToWorld(matrix)
            matrixKey = tuple(matrix.GetElement(r, c) for r in range(4) for c in range(4))
        key = (modelNode.GetID(), polyData.GetMTime() if polyData else 0, matrixKey)
        if self._surfaceLocatorCache is None or self._surfaceLocatorCache[0] != key:
            locator = vtk.vtkCellLocator()
            locator.SetDataSet(self.getWorldPolyData(modelNode))
            locator.BuildLocator()
            self._surfaceLocatorCache = (key, locator)
        return self._surfaceLocatorCache[1]

    #
    # Cutting sheet preview (MRML)
    #

    def computeModelExtent(self, modelNode: vtkMRMLModelNode, options: CutOptions) -> float:
        """Sheet extent for a model: options.extension if set, otherwise its world bounding-box diagonal.

        Uses the node's world (RAS) bounds, so it is cheap enough for live preview.
        """
        if options.extension > 0:
            return float(options.extension)
        bounds = np.zeros(6)
        modelNode.GetRASBounds(bounds)
        diagonal = float(np.linalg.norm(bounds[1::2] - bounds[0::2]))
        if not diagonal > 0:
            raise ValueError(_("The selected model is empty."))
        return diagonal

    def buildSheetForParameters(self, parameterNode: OsteotomyCutsParameterNode) -> vtk.vtkPolyData:
        """Build the cutting sheet from the parameter node's model, cut path, direction and options.

        :raises ValueError: if the inputs are invalid.
        """
        reason = self.validateInputs(parameterNode)
        if reason:
            raise ValueError(reason)
        curveNode, options = parameterNode.cutCurve, parameterNode.options
        return self.buildSheetPolyData(self.getPathPoints(curveNode), self.resolveDirection(parameterNode),
                                       self.computeModelExtent(parameterNode.inputModel, options),
                                       closed=self.isClosedCurve(curveNode),
                                       depth=options.depth if options.depth > 0 else None)

    def updateSheetModel(self, parameterNode: OsteotomyCutsParameterNode) -> Optional[vtkMRMLModelNode]:
        """Show the cutting sheet for the current inputs (live preview).

        Creates the "CuttingSheet" model node on first use (semi-transparent red, hidden from
        node selectors, not selectable so points are not placed on it) and stores it in the
        parameter node. The sheet is hidden if live preview is off or the inputs are invalid.

        :return: the sheet model node, or None if no sheet is shown.
        """
        if not parameterNode.livePreview:
            self.hideSheetModel(parameterNode)
            return None
        try:
            sheet = self.buildSheetForParameters(parameterNode)
        except ValueError:
            self.hideSheetModel(parameterNode)
            return None

        sheetNode = parameterNode.sheetModel
        if sheetNode is None:
            sheetNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "CuttingSheet")
            sheetNode.SetHideFromEditors(True)
            sheetNode.SetSelectable(False)
            sheetNode.CreateDefaultDisplayNodes()
            displayNode = sheetNode.GetDisplayNode()
            displayNode.SetColor(0.9, 0.2, 0.2)
            displayNode.SetOpacity(0.4)
            displayNode.SetBackfaceCulling(False)
            displayNode.SetVisibility2D(True)
            parameterNode.sheetModel = sheetNode
        sheetNode.SetAndObservePolyData(sheet)
        sheetNode.GetDisplayNode().SetVisibility(True)
        return sheetNode

    def hideSheetModel(self, parameterNode: OsteotomyCutsParameterNode) -> None:
        """Hide the cutting sheet preview, if there is one."""
        sheetNode = parameterNode.sheetModel
        if sheetNode is not None and sheetNode.GetDisplayNode() is not None:
            sheetNode.GetDisplayNode().SetVisibility(False)

    #
    # Cut results (MRML)
    #

    def applyCut(self, parameterNode: OsteotomyCutsParameterNode,
                 progressCallback: Optional[ProgressCallback] = None) -> list[vtkMRMLModelNode]:
        """Cut the input model with the cut path and create the fragment models.

        The result this same curve produced before is replaced; results of other curves are
        never touched. The fragments are computed first, so on failure the previous result
        stays as it was.

        :param parameterNode: module parameters (input model, cut path, direction, options).
        :param progressCallback: called with (percent, message) between stages.
        :return: the new fragment model nodes, largest first.
        :raises ValueError: if inputs are invalid, the sheet does not cut the model, or later
            cuts depend on the previous result of this curve.
        """
        reason = self.validateInputs(parameterNode)
        if reason:
            raise ValueError(reason)
        inputModel, curveNode = parameterNode.inputModel, parameterNode.cutCurve
        self._checkNoDependentCuts(curveNode)

        report = progressCallback or (lambda percent, message: None)
        report(0, _("Preparing..."))
        polyData = self.getWorldPolyData(inputModel)
        sheet = self.buildSheetForParameters(parameterNode)
        fragments = self.cutPolyData(polyData, [sheet], parameterNode.options, progressCallback)
        if len(fragments) < 2:
            raise ValueError(_("The cutting sheet does not divide the model. Check the cut path and direction."))

        report(90, _("Creating fragment models..."))
        self.removeCutResult(curveNode)
        nodes = self.createFragmentNodes(fragments, inputModel, curveNode)
        report(100, _("Done."))
        return nodes

    def createFragmentNodes(self, fragments: list[vtk.vtkPolyData], inputModel: vtkMRMLModelNode,
                            curveNode: vtkMRMLMarkupsCurveNode) -> list[vtkMRMLModelNode]:
        """Create one model node per fragment and hide the input model.

        Nodes are named <InputModelName>_<CurveName>_<N>, coloured distinctly, placed in the
        subject hierarchy folder "<InputModelName>_<CurveName>" next to the input model, and
        linked to the curve and input model with node references.

        :param fragments: fragment meshes in world coordinates.
        :param inputModel: the model that was cut (hidden, not modified).
        :param curveNode: the cut path that produced the fragments.
        :return: the new model nodes, in the order of ``fragments``.
        """
        baseName = f"{inputModel.GetName()}_{curveNode.GetName()}"
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        parentItem = shNode.GetItemParent(shNode.GetItemByDataNode(inputModel))
        folderItem = shNode.CreateFolderItem(parentItem, baseName)

        curveNode.SetNodeReferenceID(INPUT_REFERENCE_ROLE, inputModel.GetID())
        nodes = []
        for index, fragment in enumerate(fragments):
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{baseName}_{index + 1}")
            node.SetAndObservePolyData(fragment)
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetColor(self.fragmentColour(index))
            shNode.SetItemParent(shNode.GetItemByDataNode(node), folderItem)
            node.SetNodeReferenceID(CURVE_REFERENCE_ROLE, curveNode.GetID())
            node.SetNodeReferenceID(INPUT_REFERENCE_ROLE, inputModel.GetID())
            curveNode.AddNodeReferenceID(FRAGMENT_REFERENCE_ROLE, node.GetID())
            nodes.append(node)

        if inputModel.GetDisplayNode():
            inputModel.GetDisplayNode().SetVisibility(False)
        return nodes

    def getCurveResult(self, curveNode: vtkMRMLMarkupsCurveNode) -> list[vtkMRMLModelNode]:
        """Return the fragment models this curve produced (empty if it has not cut anything)."""
        if curveNode is None:
            return []
        nodes = (curveNode.GetNthNodeReference(FRAGMENT_REFERENCE_ROLE, i)
                 for i in range(curveNode.GetNumberOfNodeReferences(FRAGMENT_REFERENCE_ROLE)))
        return [node for node in nodes if node is not None]

    def getDependentCurves(self, curveNode: vtkMRMLMarkupsCurveNode) -> list[vtkMRMLMarkupsCurveNode]:
        """Return the curves whose current result was cut from one of this curve's fragments."""
        fragmentIds = {node.GetID() for node in self.getCurveResult(curveNode)}
        dependents = []
        for other in slicer.util.getNodesByClass("vtkMRMLMarkupsCurveNode"):
            if other.GetID() == curveNode.GetID() or not self.getCurveResult(other):
                continue
            if other.GetNodeReferenceID(INPUT_REFERENCE_ROLE) in fragmentIds:
                dependents.append(other)
        return dependents

    def removeCutResult(self, curveNode: vtkMRMLMarkupsCurveNode) -> None:
        """Undo one curve's cut: delete its fragments and folder and show its input model again.

        Does nothing if the curve has no result.

        :raises ValueError: if other curves have cut this curve's fragments (undo those first).
        """
        fragments = self.getCurveResult(curveNode)
        if not fragments:
            return
        self._checkNoDependentCuts(curveNode)

        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        folderItems = {shNode.GetItemParent(shNode.GetItemByDataNode(node)) for node in fragments}
        for node in fragments:
            slicer.mrmlScene.RemoveNode(node)
        for folderItem in folderItems:
            if folderItem != shNode.GetSceneItemID() and shNode.GetNumberOfItemChildren(folderItem) == 0:
                shNode.RemoveItem(folderItem)

        inputModel = curveNode.GetNodeReference(INPUT_REFERENCE_ROLE)
        if inputModel is not None and inputModel.GetDisplayNode():
            inputModel.GetDisplayNode().SetVisibility(True)
        curveNode.RemoveNodeReferenceIDs(FRAGMENT_REFERENCE_ROLE)
        curveNode.RemoveNodeReferenceIDs(INPUT_REFERENCE_ROLE)

    def mergeFragments(self, fragmentNodes: list[vtkMRMLModelNode]) -> vtkMRMLModelNode:
        """Join two or more fragments of the same cut into one model.

        The merged model keeps the name and colour of the lowest-numbered fragment; the other
        fragments are removed. Undo of the cut still removes everything. This is the Phase 1
        workaround for unwanted through-cuts (e.g. of the contralateral side).

        :param fragmentNodes: fragment models, all produced by the same cut path.
        :return: the merged model node.
        :raises ValueError: fewer than 2 fragments, fragments of different cuts, different
            parent transforms, or a fragment that has been cut further.
        """
        nodes = list({node.GetID(): node for node in fragmentNodes}.values())  # unique, order kept
        if len(nodes) < 2:
            raise ValueError(_("Select at least 2 fragments to merge."))
        curveIds = {node.GetNodeReferenceID(CURVE_REFERENCE_ROLE) for node in nodes}
        if len(curveIds) != 1 or None in curveIds:
            raise ValueError(_("Only fragments produced by the same cut path can be merged."))
        curveNode = nodes[0].GetNodeReference(CURVE_REFERENCE_ROLE)
        if len({node.GetTransformNodeID() for node in nodes}) != 1:
            raise ValueError(_("The fragments are under different transforms."))
        nodeIds = {node.GetID() for node in nodes}
        for other in self.getDependentCurves(curveNode):
            if other.GetNodeReferenceID(INPUT_REFERENCE_ROLE) in nodeIds:
                raise ValueError(_("Fragment {fragment} has been cut by {curve}. Undo that cut first.").format(
                    fragment=other.GetNodeReference(INPUT_REFERENCE_ROLE).GetName(), curve=other.GetName()))

        resultIds = [node.GetID() for node in self.getCurveResult(curveNode)]
        nodes.sort(key=lambda node: resultIds.index(node.GetID()))
        keep, others = nodes[0], nodes[1:]
        keep.SetAndObservePolyData(self.mergePolyData([node.GetPolyData() for node in nodes]))
        for node in others:
            self._removeReference(curveNode, FRAGMENT_REFERENCE_ROLE, node)
            slicer.mrmlScene.RemoveNode(node)
        return keep

    @staticmethod
    def fragmentColour(index: int) -> tuple[float, float, float]:
        """Return a distinct display colour for the fragment with the given 0-based index."""
        return FRAGMENT_COLOURS[index % len(FRAGMENT_COLOURS)]

    def _checkNoDependentCuts(self, curveNode: vtkMRMLMarkupsCurveNode) -> None:
        """Raise ValueError if other curves have cut this curve's fragments."""
        dependents = self.getDependentCurves(curveNode)
        if dependents:
            raise ValueError(_("Fragments of {curve} have been cut further by {others}. Undo those cuts first.").format(
                curve=curveNode.GetName(), others=", ".join(other.GetName() for other in dependents)))

    @staticmethod
    def _removeReference(referencingNode, role: str, referencedNode) -> None:
        """Remove one node reference of the given role pointing to referencedNode."""
        for i in reversed(range(referencingNode.GetNumberOfNodeReferences(role))):
            if referencingNode.GetNthNodeReferenceID(role, i) == referencedNode.GetID():
                referencingNode.RemoveNthNodeReferenceID(role, i)

    @staticmethod
    def _normalised(vector: np.ndarray, errorMessage: str) -> np.ndarray:
        """Return vector / |vector|, or raise ValueError(errorMessage) for a zero vector."""
        length = float(np.linalg.norm(vector))
        if length < 1e-9:
            raise ValueError(errorMessage)
        return np.asarray(vector, dtype=float) / length

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
        # Phase 2 defaults reproduce the Phase 1 zero-width through-cut
        self.assertEqual(parameterNode.options.kerfWidth, 0.0)
        self.assertEqual(parameterNode.options.depth, 0.0)
        self.assertTrue(parameterNode.options.capCutFaces)
        self.assertEqual(parameterNode.options.refineEdgeLength, 0.0)

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

    #
    # Step 4: MRML results
    #

    X_CUT_PATH = [[1.3, -40.0, 50.0], [1.3, 40.0, 50.0]]  # across the top of the test box, at x = 1.3
    DOWN = (0.0, 0.0, -1.0)

    @staticmethod
    def _addModel(polyData: vtk.vtkPolyData, name: str) -> vtkMRMLModelNode:
        node = slicer.modules.models.logic().AddModel(polyData)
        node.SetName(name)
        return node

    @staticmethod
    def _addCurve(points, name: str, closed: bool = False) -> vtkMRMLMarkupsCurveNode:
        className = "vtkMRMLMarkupsClosedCurveNode" if closed else "vtkMRMLMarkupsCurveNode"
        node = slicer.mrmlScene.AddNewNodeByClass(className, name)
        node.SetCurveTypeToLinear()
        slicer.util.updateMarkupsControlPointsFromArray(node, np.array(points, dtype=float))
        return node

    @staticmethod
    def _configure(model, curve, direction=DOWN) -> OsteotomyCutsParameterNode:
        parameterNode = OsteotomyCutsLogic().getParameterNode()
        parameterNode.inputModel = model
        parameterNode.cutCurve = curve
        parameterNode.directionMode = DirectionMode.VIEW
        parameterNode.viewDirection = tuple(direction)
        return parameterNode

    def _cutModel(self, model, curve, direction=DOWN) -> list:
        return OsteotomyCutsLogic().applyCut(self._configure(model, curve, direction))

    @staticmethod
    def _ids(nodes) -> list:
        return [node.GetID() for node in nodes]

    @staticmethod
    def _isVisible(model) -> bool:
        return bool(model.GetDisplayNode().GetVisibility())

    def test_mrml_output(self):
        """Fragments are named, coloured, grouped, linked by references; the input is hidden only."""
        logic = OsteotomyCutsLogic()
        model = self._addModel(self._box(), "Box")
        pointsBefore = self._points(model.GetPolyData()).copy()
        curve = self._addCurve(self.X_CUT_PATH, "CutA")
        nodes = self._cutModel(model, curve)

        self.assertEqual([node.GetName() for node in nodes], ["Box_CutA_1", "Box_CutA_2"])
        colours = {tuple(node.GetDisplayNode().GetColor()) for node in nodes}
        self.assertEqual(len(colours), 2)

        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        folders = {shNode.GetItemParent(shNode.GetItemByDataNode(node)) for node in nodes}
        self.assertEqual(len(folders), 1)
        folder = folders.pop()
        self.assertEqual(shNode.GetItemName(folder), "Box_CutA")
        self.assertEqual(shNode.GetItemParent(folder), shNode.GetItemParent(shNode.GetItemByDataNode(model)))

        self.assertEqual(self._ids(logic.getCurveResult(curve)), self._ids(nodes))
        self.assertEqual(curve.GetNodeReferenceID(INPUT_REFERENCE_ROLE), model.GetID())
        for node in nodes:
            self.assertEqual(node.GetNodeReferenceID(CURVE_REFERENCE_ROLE), curve.GetID())
            self.assertEqual(node.GetNodeReferenceID(INPUT_REFERENCE_ROLE), model.GetID())
            self.assertIsNone(node.GetParentTransformNode())
            self.assertTrue(self._isVisible(node))

        self.assertFalse(self._isVisible(model))
        self.assertIsNotNone(slicer.mrmlScene.GetNodeByID(model.GetID()))
        np.testing.assert_array_equal(self._points(model.GetPolyData()), pointsBefore)

    def test_sequentialCut(self):
        """Cutting a fragment gives sub-fragments and leaves the other fragments alone."""
        logic = OsteotomyCutsLogic()
        model = self._addModel(self._box(), "Box")
        curveA = self._addCurve(self.X_CUT_PATH, "CutA")
        fragmentsA = self._cutModel(model, curveA)
        right = next(node for node in fragmentsA if node.GetPolyData().GetBounds()[0] > 0.0)
        left = next(node for node in fragmentsA if node is not right)

        curveB = self._addCurve([[10.0, 2.7, 50.0], [40.0, 2.7, 50.0]], "CutB")
        fragmentsB = self._cutModel(right, curveB)

        self.assertEqual(len(fragmentsB), 2)
        for node in fragmentsB:
            self.assertTrue(node.GetName().startswith(f"{right.GetName()}_CutB_"))
            self.assertEqual(node.GetNodeReferenceID(INPUT_REFERENCE_ROLE), right.GetID())
        self.assertFalse(self._isVisible(right))
        self.assertTrue(self._isVisible(left))
        self.assertEqual(self._ids(logic.getCurveResult(curveA)), self._ids(fragmentsA))
        self.assertEqual(self._ids(logic.getDependentCurves(curveA)), [curveB.GetID()])

    def test_reapplyPerCurve(self):
        """Re-applying a curve replaces only its own result."""
        logic = OsteotomyCutsLogic()
        model1 = self._addModel(self._box(), "Box1")
        model2 = self._addModel(self._box(), "Box2")
        curveA = self._addCurve(self.X_CUT_PATH, "CutA")
        curveB = self._addCurve(self.X_CUT_PATH, "CutB")
        oldA = self._ids(self._cutModel(model1, curveA))
        resultB = self._ids(self._cutModel(model2, curveB))

        curveA.SetNthControlPointPositionWorld(0, 5.3, -40.0, 50.0)
        curveA.SetNthControlPointPositionWorld(1, 5.3, 40.0, 50.0)
        newA = self._cutModel(model1, curveA)

        self.assertEqual(len(newA), 2)
        self.assertEqual([node.GetName() for node in newA], ["Box1_CutA_1", "Box1_CutA_2"])
        for nodeId in oldA:
            self.assertIsNone(slicer.mrmlScene.GetNodeByID(nodeId))
        self.assertEqual(self._ids(logic.getCurveResult(curveB)), resultB)
        for nodeId in resultB:
            self.assertIsNotNone(slicer.mrmlScene.GetNodeByID(nodeId))
        self.assertFalse(self._isVisible(model1))
        self.assertFalse(self._isVisible(model2))

    def test_dependentBlocks(self):
        """A cut whose fragments were cut further cannot be undone, re-applied or merged."""
        logic = OsteotomyCutsLogic()
        model = self._addModel(self._box(), "Box")
        curveA = self._addCurve(self.X_CUT_PATH, "CutA")
        fragmentsA = self._cutModel(model, curveA)
        right = next(node for node in fragmentsA if node.GetPolyData().GetBounds()[0] > 0.0)
        curveB = self._addCurve([[10.0, 2.7, 50.0], [40.0, 2.7, 50.0]], "CutB")
        self._cutModel(right, curveB)

        with self.assertRaises(ValueError):
            logic.removeCutResult(curveA)
        with self.assertRaises(ValueError):
            self._cutModel(model, curveA)
        with self.assertRaises(ValueError):
            logic.mergeFragments(fragmentsA)
        self.assertEqual(self._ids(logic.getCurveResult(curveA)), self._ids(fragmentsA))

        logic.removeCutResult(curveB)
        logic.removeCutResult(curveA)
        self.assertEqual(logic.getCurveResult(curveA), [])
        self.assertTrue(self._isVisible(model))

    def test_undo(self):
        """Undo removes the fragments and their folder and shows the input again."""
        logic = OsteotomyCutsLogic()
        model = self._addModel(self._box(), "Box")
        curve = self._addCurve(self.X_CUT_PATH, "CutA")
        nodeIds = self._ids(self._cutModel(model, curve))

        logic.removeCutResult(curve)

        for nodeId in nodeIds:
            self.assertIsNone(slicer.mrmlScene.GetNodeByID(nodeId))
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        self.assertEqual(shNode.GetItemChildWithName(shNode.GetSceneItemID(), "Box_CutA"), 0)
        self.assertTrue(self._isVisible(model))
        self.assertEqual(logic.getCurveResult(curve), [])
        self.assertIsNone(curve.GetNodeReferenceID(INPUT_REFERENCE_ROLE))
        logic.removeCutResult(curve)  # nothing left to undo: no error

    def test_mergeFragments(self):
        """Merging keeps the lowest-numbered name and colour, and undo still removes everything."""
        logic = OsteotomyCutsLogic()
        radius, circleRadius = 30.0, 10.0
        angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        z = np.sqrt(radius ** 2 - circleRadius ** 2)
        circle = np.column_stack([circleRadius * np.cos(angles), circleRadius * np.sin(angles),
                                  np.full_like(angles, z)])
        model = self._addModel(self._sphere(radius), "Sphere")
        curve = self._addCurve(circle, "Ring", closed=True)
        band, cap1, cap2 = self._cutModel(model, curve, direction=(0.0, 0.0, 1.0))
        capPoints = cap1.GetPolyData().GetNumberOfPoints() + cap2.GetPolyData().GetNumberOfPoints()
        cap1Colour = tuple(cap1.GetDisplayNode().GetColor())
        cap2Id = cap2.GetID()

        with self.assertRaises(ValueError):
            logic.mergeFragments([cap1])
        otherModel = self._addModel(self._box(), "Box")
        otherFragments = self._cutModel(otherModel, self._addCurve(self.X_CUT_PATH, "CutA"))
        with self.assertRaises(ValueError):
            logic.mergeFragments([cap1, otherFragments[0]])

        merged = logic.mergeFragments([cap2, cap1])

        self.assertEqual(merged.GetID(), cap1.GetID())
        self.assertEqual(merged.GetName(), "Sphere_Ring_2")
        self.assertEqual(tuple(merged.GetDisplayNode().GetColor()), cap1Colour)
        self.assertEqual(merged.GetPolyData().GetNumberOfPoints(), capPoints)
        self.assertIsNone(slicer.mrmlScene.GetNodeByID(cap2Id))
        self.assertEqual(self._ids(logic.getCurveResult(curve)), [band.GetID(), merged.GetID()])

        logic.removeCutResult(curve)
        self.assertIsNone(slicer.mrmlScene.GetNodeByID(merged.GetID()))
        self.assertIsNone(slicer.mrmlScene.GetNodeByID(band.GetID()))
        self.assertTrue(self._isVisible(model))

    def test_transformedInput(self):
        """A model under a linear transform is cut in world coordinates."""
        model = self._addModel(self._box(), "Box")
        transform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
        matrix = vtk.vtkMatrix4x4()
        matrix.SetElement(0, 3, 100.0)
        transform.SetMatrixTransformToParent(matrix)
        model.SetAndObserveTransformNodeID(transform.GetID())
        curve = self._addCurve([[101.3, -40.0, 50.0], [101.3, 40.0, 50.0]], "CutA")

        nodes = self._cutModel(model, curve)

        self.assertEqual(len(nodes), 2)
        boundsList = sorted((node.GetPolyData().GetBounds() for node in nodes), key=lambda b: b[0])
        self.assertAlmostEqual(boundsList[0][0], 50.0, places=3)
        self.assertAlmostEqual(boundsList[0][1], 101.3, places=3)
        self.assertAlmostEqual(boundsList[1][0], 101.3, places=3)
        self.assertAlmostEqual(boundsList[1][1], 150.0, places=3)

    def test_directions(self):
        """Directions come from a markups line or a 3D view camera."""
        logic = OsteotomyCutsLogic()
        line = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
        slicer.util.updateMarkupsControlPointsFromArray(line, np.array([[1.0, 2.0, 3.0], [1.0, 2.0, -7.0]]))
        np.testing.assert_allclose(logic.directionFromLine(line), [0.0, 0.0, -1.0], atol=1e-9)

        viewNode = slicer.vtkMRMLViewNode()
        viewNode.SetLayoutName("OsteotomyCutsTestView")  # a view needs a layout name to own a camera
        slicer.mrmlScene.AddNode(viewNode)
        cameraNode = slicer.modules.cameras.logic().GetViewActiveCameraNode(viewNode)
        if cameraNode is None:
            cameraNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLCameraNode")
            cameraNode.SetLayoutName(viewNode.GetLayoutName())
        cameraNode.SetPosition(0.0, 100.0, 0.0)
        cameraNode.SetFocalPoint(0.0, 0.0, 0.0)
        np.testing.assert_allclose(logic.directionFromView(viewNode), [0.0, -1.0, 0.0], atol=1e-9)

        parameterNode = logic.getParameterNode()
        logic.captureViewDirection(parameterNode, viewNode)
        np.testing.assert_allclose(parameterNode.viewDirection, [0.0, -1.0, 0.0], atol=1e-9)

    def test_validateInputs(self):
        """validateInputs says what is missing, and None once a cut can run."""
        logic = OsteotomyCutsLogic()
        parameterNode = logic.getParameterNode()
        self.assertIn("model", logic.validateInputs(parameterNode))

        model = self._addModel(self._box(), "Box")
        parameterNode.inputModel = model
        self.assertIn("cut path", logic.validateInputs(parameterNode))

        curve = self._addCurve([[1.3, -40.0, 50.0]], "CutA")
        parameterNode.cutCurve = curve
        self.assertIn("2 points", logic.validateInputs(parameterNode))

        curve.AddControlPointWorld(1.3, 40.0, 50.0)
        self.assertIn("view direction", logic.validateInputs(parameterNode))

        parameterNode.viewDirection = self.DOWN
        self.assertIsNone(logic.validateInputs(parameterNode))

        parameterNode.directionMode = DirectionMode.LINE
        self.assertIn("direction line", logic.validateInputs(parameterNode))
        line = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
        slicer.util.updateMarkupsControlPointsFromArray(line, np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 0.0]]))
        parameterNode.directionLine = line
        self.assertIsNone(logic.validateInputs(parameterNode))

        # A fragment cannot be cut again by the curve that produced it
        fragment = logic.applyCut(parameterNode)[0]
        parameterNode.inputModel = fragment
        self.assertIn("produced by this cut path", logic.validateInputs(parameterNode))
        with self.assertRaises(ValueError):
            logic.applyCut(parameterNode)

    def test_failedCutKeepsResult(self):
        """If a re-applied cut fails (sheet misses the model), the previous result stays."""
        logic = OsteotomyCutsLogic()
        model = self._addModel(self._box(), "Box")
        curve = self._addCurve(self.X_CUT_PATH, "CutA")
        nodeIds = self._ids(self._cutModel(model, curve))

        curve.SetNthControlPointPositionWorld(0, 200.0, -40.0, 50.0)
        curve.SetNthControlPointPositionWorld(1, 200.0, 40.0, 50.0)
        with self.assertRaises(ValueError):
            self._cutModel(model, curve)

        self.assertEqual(self._ids(logic.getCurveResult(curve)), nodeIds)
        self.assertFalse(self._isVisible(model))

    #
    # Step 5: preview, snapping, scene round trip
    #

    def test_snap(self):
        """Points off the surface are moved onto it; points on it are left alone."""
        logic = OsteotomyCutsLogic()
        model = self._addModel(self._sphere(30.0), "Sphere")
        curve = self._addCurve([[35.0, 0.0, 0.0], [0.0, 25.0, 0.0], [0.0, 0.0, 34.0]], "CutA")

        self.assertEqual(logic.snapCurveToSurface(curve, model), 3)
        radii = np.linalg.norm(slicer.util.arrayFromMarkupsControlPoints(curve, world=True), axis=1)
        np.testing.assert_allclose(radii, 30.0, atol=0.1)
        self.assertEqual(logic.snapCurveToSurface(curve, model), 0)

    def test_snap_transformedModel(self):
        """Snapping uses the model surface in world coordinates."""
        logic = OsteotomyCutsLogic()
        model = self._addModel(self._sphere(30.0), "Sphere")
        transform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
        matrix = vtk.vtkMatrix4x4()
        matrix.SetElement(1, 3, 50.0)
        transform.SetMatrixTransformToParent(matrix)
        model.SetAndObserveTransformNodeID(transform.GetID())
        curve = self._addCurve([[0.0, 50.0, 40.0], [0.0, 90.0, 0.0]], "CutA")

        logic.snapCurveToSurface(curve, model)

        points = slicer.util.arrayFromMarkupsControlPoints(curve, world=True)
        np.testing.assert_allclose(np.linalg.norm(points - [0.0, 50.0, 0.0], axis=1), 30.0, atol=0.1)

    def test_sheetPreview(self):
        """The preview sheet follows the inputs, and is hidden when they are invalid or preview is off."""
        logic = OsteotomyCutsLogic()
        model = self._addModel(self._box(), "Box")
        curve = self._addCurve(self.X_CUT_PATH, "CutA")
        parameterNode = self._configure(model, curve)

        sheetNode = logic.updateSheetModel(parameterNode)
        self.assertIsNotNone(sheetNode)
        self.assertEqual(parameterNode.sheetModel.GetID(), sheetNode.GetID())
        self.assertTrue(sheetNode.GetHideFromEditors())
        self.assertFalse(sheetNode.GetSelectable())
        self.assertTrue(sheetNode.GetDisplayNode().GetVisibility())
        self.assertEqual(sheetNode.GetPolyData().GetNumberOfPoints(), 8)
        self.assertAlmostEqual(sheetNode.GetPolyData().GetBounds()[0], 1.3, places=6)

        # Moving a point updates the same sheet node
        curve.SetNthControlPointPositionWorld(1, 20.0, 40.0, 50.0)
        self.assertEqual(logic.updateSheetModel(parameterNode).GetID(), sheetNode.GetID())
        self.assertGreater(sheetNode.GetPolyData().GetBounds()[1], 20.0)

        # The preview and the cut use the same sheet
        np.testing.assert_allclose(logic.buildSheetForParameters(parameterNode).GetBounds(),
                                   sheetNode.GetPolyData().GetBounds())

        parameterNode.viewDirection = NOT_CAPTURED
        self.assertIsNone(logic.updateSheetModel(parameterNode))
        self.assertFalse(sheetNode.GetDisplayNode().GetVisibility())

        parameterNode.viewDirection = self.DOWN
        parameterNode.livePreview = False
        self.assertIsNone(logic.updateSheetModel(parameterNode))
        self.assertFalse(sheetNode.GetDisplayNode().GetVisibility())
        self.assertEqual(len(slicer.util.getNodes("CuttingSheet*")), 1)

    def test_sceneRoundTrip(self):
        """Parameters, the captured direction and cut results survive saving and reloading a scene."""
        import os
        logic = OsteotomyCutsLogic()
        model = self._addModel(self._box(), "Box")
        curve = self._addCurve(self.X_CUT_PATH, "CutA")
        parameterNode = self._configure(model, curve, direction=(0.0, 0.6, -0.8))
        parameterNode.options.minFragmentFraction = 0.02
        logic.updateSheetModel(parameterNode)
        fragmentNames = [node.GetName() for node in logic.applyCut(parameterNode)]

        scenePath = os.path.join(slicer.app.temporaryPath, "OsteotomyCutsRoundTrip.mrb")
        try:
            self.assertTrue(slicer.util.saveScene(scenePath))
            slicer.mrmlScene.Clear()
            slicer.util.loadScene(scenePath)
        finally:
            if os.path.exists(scenePath):
                os.remove(scenePath)

        parameterNode = OsteotomyCutsLogic().getParameterNode()
        self.assertEqual(parameterNode.inputModel.GetName(), "Box")
        self.assertEqual(parameterNode.cutCurve.GetName(), "CutA")
        np.testing.assert_allclose(parameterNode.viewDirection, (0.0, 0.6, -0.8), atol=1e-9)
        self.assertAlmostEqual(parameterNode.options.minFragmentFraction, 0.02)
        self.assertIsNotNone(parameterNode.sheetModel)

        logic = OsteotomyCutsLogic()
        curve = parameterNode.cutCurve
        self.assertEqual([node.GetName() for node in logic.getCurveResult(curve)], fragmentNames)
        self.assertEqual(curve.GetNodeReference(INPUT_REFERENCE_ROLE).GetName(), "Box")
        self.assertFalse(self._isVisible(parameterNode.inputModel))

        logic.removeCutResult(curve)
        self.assertEqual(logic.getCurveResult(curve), [])
        self.assertTrue(self._isVisible(parameterNode.inputModel))

    #
    # Phase 2 step 1: depth, options
    #

    def test_sheet_depth(self):
        """With a depth, the sheet reaches outward by the extent but inward only by the depth."""
        logic = OsteotomyCutsLogic()
        extent, depth = 50.0, 8.0
        path = np.array([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        down = np.array([0.0, 0.0, -1.0])
        sheet = logic.buildSheetPolyData(path, down, extent, depth=depth)

        points = self._points(sheet)
        np.testing.assert_allclose(points[0::2, 2], extent)  # outward vertices (against d)
        np.testing.assert_allclose(points[1::2, 2], -depth)  # inward vertices (along d)
        self.assertAlmostEqual(sheet.GetBounds()[4], -depth)

        # One depth per path point; end extensions take the depth of their end point
        sheet = logic.buildSheetPolyData(path, down, extent, depth=np.array([2.0, 6.0]))
        np.testing.assert_allclose(self._points(sheet)[1::2, 2], [-2.0, -2.0, -6.0, -6.0])

        # No depth is a through-cut, as in Phase 1
        np.testing.assert_allclose(self._points(logic.buildSheetPolyData(path, down, extent))[1::2, 2], -extent)

    def test_sheet_depth_errors(self):
        """Zero, negative or wrongly sized depths raise ValueError."""
        logic = OsteotomyCutsLogic()
        path = np.array([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        down = np.array([0.0, 0.0, -1.0])
        for depth in (0.0, -1.0, np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0])):
            with self.assertRaises(ValueError, msg=str(depth)):
                logic.buildSheetPolyData(path, down, 50.0, depth=depth)

    def test_depthOptions(self):
        """Depth needs a kerf; the preview sheet follows the depth; kerf cutting is not available yet."""
        logic = OsteotomyCutsLogic()
        model = self._addModel(self._box(), "Box")
        curve = self._addCurve(self.X_CUT_PATH, "CutA")
        parameterNode = self._configure(model, curve)

        parameterNode.options.depth = 12.0
        self.assertIn("kerf width", logic.validateInputs(parameterNode))
        self.assertIsNone(logic.updateSheetModel(parameterNode))

        parameterNode.options.kerfWidth = 1.0
        self.assertIsNone(logic.validateInputs(parameterNode))
        sheetNode = logic.updateSheetModel(parameterNode)
        self.assertAlmostEqual(sheetNode.GetPolyData().GetBounds()[4], 50.0 - 12.0, places=6)

        with self.assertRaises(ValueError):  # until Phase 2 step 3
            logic.applyCut(parameterNode)
        self.assertEqual(logic.getCurveResult(curve), [])
        self.assertTrue(self._isVisible(model))

    def test_oldSceneUpgrade(self):
        """A scene saved before newer options existed loads, keeps its values and gets defaults."""
        import os
        logic = OsteotomyCutsLogic()
        parameterNode = self._configure(self._addModel(self._box(), "Box"), self._addCurve(self.X_CUT_PATH, "CutA"))
        parameterNode.options.extension = 250.0
        rawNode = parameterNode.parameterNode
        del parameterNode  # the wrapper would try (and fail) to re-read the node as it is edited
        newOptions = ("options.kerfWidth", "options.depth", "options.capCutFaces", "options.refineEdgeLength")
        for name in newOptions:  # as saved by Phase 1
            rawNode.UnsetParameter(name)

        scenePath = os.path.join(slicer.app.temporaryPath, "OsteotomyCutsOldScene.mrb")
        try:
            self.assertTrue(slicer.util.saveScene(scenePath))
            slicer.mrmlScene.Clear()
            slicer.util.loadScene(scenePath)
        finally:
            if os.path.exists(scenePath):
                os.remove(scenePath)

        parameterNode = logic.getParameterNode()  # failed before the upgrade was added
        self.assertEqual(parameterNode.options.extension, 250.0)
        self.assertEqual(parameterNode.options.kerfWidth, 0.0)
        self.assertTrue(parameterNode.options.capCutFaces)
        self.assertEqual(parameterNode.cutCurve.GetName(), "CutA")
        self.assertEqual(logic.addMissingParameters(parameterNode.parameterNode), [])

    #
    # Phase 2 step 2: refinement near the sheet
    #

    @staticmethod
    def _openEdgeCount(polyData: vtk.vtkPolyData) -> int:
        """Number of boundary and non-manifold edges (0 for a closed, crack-free surface)."""
        edges = vtk.vtkFeatureEdges()
        edges.SetInputData(polyData)
        edges.BoundaryEdgesOn()
        edges.NonManifoldEdgesOn()
        edges.FeatureEdgesOff()
        edges.ManifoldEdgesOff()
        edges.Update()
        return edges.GetOutput().GetNumberOfCells()

    @staticmethod
    def _volume(polyData: vtk.vtkPolyData) -> float:
        massProperties = vtk.vtkMassProperties()
        massProperties.SetInputData(polyData)
        massProperties.Update()
        return massProperties.GetVolume()

    @staticmethod
    def _edges(polyData: vtk.vtkPolyData) -> tuple[np.ndarray, np.ndarray]:
        """Unique edges of a triangle mesh as (start ids, end ids)."""
        triangles = numpy_support.vtk_to_numpy(polyData.GetPolys().GetConnectivityArray()).reshape(-1, 3)
        edges = np.sort(np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
        edges = np.unique(edges, axis=0)
        return edges[:, 0], edges[:, 1]

    def _refinedBox(self, maxEdgeLength=1.0, bandWidth=2.0):
        logic = OsteotomyCutsLogic()
        box = self._box(level=5)  # 6 divisions per side: 16.7 mm edges
        sheet = logic.buildSheetPolyData(np.array([[1.3, -40.0, 50.0], [1.3, 40.0, 50.0]]),
                                         np.array([0.0, 0.0, -1.0]), logic.computeAutoExtent(box))
        withDistance = logic.computeSheetDistance(box, sheet)
        return logic, box, sheet, withDistance, logic.refineNearSheet(withDistance, sheet, maxEdgeLength, bandWidth)

    def test_refine_edgesNearSheet(self):
        """Edges within the band are no longer than the maximum; edges far away are untouched."""
        maxEdgeLength, bandWidth = 1.0, 2.0
        logic, box, sheet, withDistance, refined = self._refinedBox(maxEdgeLength, bandWidth)

        self.assertGreater(refined.GetNumberOfPoints(), box.GetNumberOfPoints())
        points = self._points(refined)
        start, end = self._edges(refined)
        lengths = np.linalg.norm(points[start] - points[end], axis=1)
        distances = np.abs(numpy_support.vtk_to_numpy(refined.GetPointData().GetArray("SheetDistance")))
        nearSheet = np.maximum(distances[start], distances[end]) < bandWidth
        self.assertTrue(np.any(nearSheet))
        self.assertLessEqual(lengths[nearSheet].max(), maxEdgeLength + 1e-9)
        # Edges far from the sheet are original edges (original points keep their ids)
        farAway = np.minimum(distances[start], distances[end]) > bandWidth + 25.0
        self.assertTrue(np.any(farAway))
        originalStart, originalEnd = self._edges(box)
        originalEdges = set(zip(originalStart.tolist(), originalEnd.tolist()))
        self.assertTrue(set(zip(start[farAway].tolist(), end[farAway].tolist())) <= originalEdges)

    def test_refine_conformingAndExact(self):
        """Refinement leaves no cracks, keeps the shape and orientation, and keeps original points."""
        logic, box, sheet, withDistance, refined = self._refinedBox()

        self.assertEqual(self._openEdgeCount(box), 0)
        self.assertEqual(self._openEdgeCount(refined), 0)
        self.assertAlmostEqual(self._volume(refined), self._volume(box), delta=1e-6 * self._volume(box))
        np.testing.assert_array_equal(self._points(refined)[:box.GetNumberOfPoints()], self._points(box))
        # Distances at new points are exact, not interpolated
        exact = logic.computeSheetDistance(refined, sheet)
        np.testing.assert_allclose(numpy_support.vtk_to_numpy(refined.GetPointData().GetArray("SheetDistance")),
                                   numpy_support.vtk_to_numpy(exact.GetPointData().GetArray("SheetDistance")),
                                   atol=1e-9)

    def test_refine_pointData(self):
        """Integer arrays keep their values at new points; normals stay unit length."""
        logic = OsteotomyCutsLogic()
        sphere = self._sphere(30.0, resolution=16)
        labelled = logic.labelEnclosedComponents(sphere, 0.001)
        sheet = logic.buildSheetPolyData(np.array([[1.3, -40.0, 29.0], [1.3, 40.0, 29.0]]),
                                         np.array([0.0, 0.0, -1.0]), 100.0)
        refined = logic.refineNearSheet(logic.computeSheetDistance(labelled, sheet), sheet, 0.5, 1.0)

        componentIds = numpy_support.vtk_to_numpy(refined.GetPointData().GetArray("ComponentId"))
        self.assertEqual(set(componentIds.tolist()), {0})
        normals = numpy_support.vtk_to_numpy(refined.GetPointData().GetNormals())
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)

    def test_refine_noopAndErrors(self):
        """Short edges are left alone; invalid lengths or too many points raise ValueError."""
        logic, box, sheet, withDistance, refined = self._refinedBox(maxEdgeLength=50.0)
        self.assertEqual(refined.GetNumberOfPoints(), box.GetNumberOfPoints())
        with self.assertRaises(ValueError):
            logic.refineNearSheet(withDistance, sheet, 0.0, 1.0)
        logic.MAX_REFINED_POINTS = box.GetNumberOfPoints() + 10
        with self.assertRaises(ValueError):
            logic.refineNearSheet(withDistance, sheet, 0.01, 1.0)

    def test_cut_withRefinement(self):
        """A zero-kerf cut with refinement gives the same fragments, with short edges at the cut."""
        options = CutOptions()
        options.refineEdgeLength = 1.0
        x = 1.3
        fragments = self._cut(self._box(level=5), [[[x, -40.0, 50.0], [x, 40.0, 50.0]]], [0.0, 0.0, -1.0],
                              options=options)

        self.assertEqual(len(fragments), 2)
        boundsList = sorted((fragment.GetBounds() for fragment in fragments), key=lambda b: b[0])
        self.assertAlmostEqual(boundsList[0][1], x, places=6)
        self.assertAlmostEqual(boundsList[1][0], x, places=6)
        for fragment in fragments:
            points = self._points(fragment)
            start, end = self._edges(fragment)
            atCut = (np.abs(points[start, 0] - x) < 1e-6) & (np.abs(points[end, 0] - x) < 1e-6)
            lengths = np.linalg.norm(points[start] - points[end], axis=1)
            self.assertLessEqual(lengths[atCut].max(), 1.0 + 1e-6)
