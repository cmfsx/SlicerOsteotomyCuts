import enum
import json
import logging
import re
import time
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
    vtkMRMLMarkupsFiducialNode,
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

The red lines of the preview show every place the cut comes out of the bone. If they appear
where bone must stay intact (e.g. the skull base behind a maxilla), limit the cut:
"Cut depth" stops it inside the bone, and "Past line ends" stops it a few millimetres beyond
the first and last points of the cut path, in a gap between bones (the cut must leave the bone
there).

Several cut paths can be cut together in one step as one osteotomy ("Further lines"). They are
cut in order; each later line only cuts on its own side of the earlier lines, so it stops where
it meets them. Each line keeps its own direction and saw settings: select it as the cut path to
see or change them.

Example, Le Fort I on a skull model (two lines):
1. Horizontal line: place the cut path points from one zygomatic buttress round the anterior
maxilla to the other, above the tooth apices. Look at the skull from the front and capture the
view direction (the cut runs backwards). Set the cut depth to about 45-55 mm (to the pterygoid
plates), "Past line ends" to 5-10 mm, and a kerf of 0.5-1.0 mm.
2. Pterygomaxillary line: create a second cut path and place its points on the side of the
maxilla behind the tuberosity, from the horizontal line downwards. Look at the skull from the
side and capture the view direction (it cuts both sides). Leave the cut depth and "Past line
ends" at 0: it stops at the horizontal cut, so the bone above is not cut.
3. Select the horizontal line again and tick the second line under "Further lines". Check that
the red lines stay on the maxilla, then apply the cut.
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

# Model node attribute ("1") marking a solid bone model: made by "Create solid bone model", or a
# capped bone segment cut from a solid model. Such models are cut as they are.
SOLID_ATTRIBUTE = "OsteotomyCuts.Solid"


class StopConflictError(ValueError):
    """A later line of an osteotomy meets bone across an earlier line, outside that line's cut."""


# Temporary integer point array: points a later osteotomy line must not remove (beyond an
# earlier line it stops at)
STOP_PROTECTED_ARRAY = "StopProtected"

# Cut path attribute holding its own settings (JSON: direction and cut options), and node
# reference roles of a cut path: its direction line, and the further lines of its osteotomy
LINE_SETTINGS_ATTRIBUTE = "OsteotomyCuts.LineSettings"
DIRECTION_LINE_REFERENCE_ROLE = "OsteotomyCuts.DirectionLine"
GROUP_LINE_REFERENCE_ROLE = "OsteotomyCuts.GroupLine"


def importNdimage():
    """Return scipy.ndimage, or None if scipy is not available (it is bundled with Slicer)."""
    try:
        from scipy import ndimage
    except ImportError:
        return None
    return ndimage

# Prefix of the temporary point arrays holding the signed distance to each kerf sheet, from
# which the side of each piece is found after the kerf is removed
SHEET_SIDE_PREFIX = "SheetSide"

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
    # Distance (mm) the cut continues past the ends of an open path; 0 = automatic (extension)
    endExtension: Annotated[float, WithinRange(0.0, 10000.0)] = 0.0
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


@dataclass
class LineSettings:
    """Direction and cut options of one osteotomy line (cut path)."""

    directionMode: DirectionMode
    viewDirection: tuple[float, float, float]
    directionLine: Optional[vtkMRMLMarkupsLineNode]
    options: CutOptions


@dataclass
class OsteotomyLine:
    """One line of an osteotomy, ready to cut: its sheet, options and the earlier lines it
    stops at ((earlier sheet, side) pairs)."""

    curve: vtkMRMLMarkupsCurveNode
    sheet: vtk.vtkPolyData
    options: CutOptions
    stops: list


@dataclass
class ModelQuality:
    """Whether a bone surface is closed, and what else may stop a cut from separating it."""

    openEdges: int  # edges with one triangle only: holes and gaps in the surface
    nonManifoldEdges: int  # edges shared by three or more triangles: surfaces crossing
    pieceCount: int  # separate connected pieces
    enclosedShellCount: int  # pieces inside another one (marrow, canals, inner cortex)

    @property
    def isClosed(self) -> bool:
        return self.openEdges == 0 and self.nonManifoldEdges == 0


@dataclass
class SegmentQuality:
    """Closedness and volume of one bone segment after a cut."""

    name: str
    isClosed: bool
    volume: float  # mm3; only meaningful for a closed segment


class StructureCategory(enum.Enum):
    """Kind of structure to protect from the cut."""

    NERVE = "nerve"
    TOOTH = "tooth"
    OTHER = "other"


# Default safe distances (mm) from the cut, and default radius (mm) of a structure traced as a curve
DEFAULT_SAFE_DISTANCES = {StructureCategory.NERVE: 2.0, StructureCategory.TOOTH: 1.0, StructureCategory.OTHER: 1.0}
DEFAULT_STRUCTURE_RADIUS = 1.5

# Model or curve attribute (JSON) marking a structure to protect, with its settings
STRUCTURE_ATTRIBUTE = "OsteotomyCuts.ProtectedStructure"

# Ends every message about structures to protect
SAFETY_NOTE = ("Distances are computed from the 3D models and depend on their accuracy; they are an aid to "
               "planning only. The surgeon remains responsible for checking them.")


@dataclass
class ProtectedStructure:
    """A model or curve (centreline) that cuts must keep clear of."""

    node: vtk.vtkObject  # vtkMRMLModelNode or vtkMRMLMarkupsCurveNode
    category: StructureCategory
    safeDistance: float  # mm
    radius: float  # mm, for a curve: the structure is a tube of this radius round it
    enabled: bool = True


class ClearanceStatus(enum.Enum):
    """How close the cut comes to a structure."""

    SAFE = "safe"  # at least the safe distance away
    TOO_CLOSE = "too close"  # closer than the safe distance
    ENTERS = "enters"  # the cut (with the blade width) enters the structure


@dataclass
class ClearanceResult:
    """Smallest distance between the cut (inside the bone) and one structure."""

    structure: ProtectedStructure
    clearance: float  # mm, minus half the blade width; inf if the cut does not reach bone
    closestPoint: Optional[np.ndarray]  # on the cut surface
    lineName: str  # the osteotomy line that comes closest
    status: ClearanceStatus


class SheetParameterisation:
    """A 2D chart of the cut surface of a sheet from buildSheetPolyData, used to cap cut faces.

    On the sheet itself, s is the length along the (extended) path and v the distance from the
    outer edge along the extrusion direction; the sheet is ruled, so (s, v) unrolls it, folds
    included, into the plane. The chart coordinates are (u, w), with u along the path and w
    across it, measured from the inner edge of the sheet (height H there): w = v - H <= 0.

    A kerf cut surface lies at distance r = kerf / 2 from the sheet: the offset surfaces on its
    + and - side (relative to the sheet normals), joined around the inner edge by a half
    cylinder, the rounded floor of a groove. Let theta be the angle of the offset direction
    from the + normal towards the inner edge (0 on the + side, pi on the - side) and t the
    distance from the inner edge along the sheet. Then w = -t on the + side, w = r * theta on
    the floor and w = r * pi + t on the - side, so one chart covers both sides and the floor
    (a groove's rim is a single loop) and its orientation is consistent across them.

    At latitude theta the surface lies at the signed offset r * cos(theta) from the sheet, and u
    is the length along it: at a fold that is convex on that side, the rounded corner of length
    |offset| * fold angle gets its own stretch of u and everything past it is shifted by that
    length; at a concave fold the offset faces meet in a crease and u jumps. For a closed sheet
    u is periodic, with a period that depends on the latitude. With r = 0 the chart is the
    sheet itself. Pure geometry, no MRML.
    """

    # A closest point this close (mm) to a fold line in s is on the fold
    FOLD_TOLERANCE = 1e-6

    def __init__(self, sheet: vtk.vtkPolyData, halfKerf: float = 0.0) -> None:
        """Derive the chart from the sheet's A, B point pairs.

        :param sheet: triangulated sheet (buildSheetPolyData output).
        :param halfKerf: offset r of the cut surface from the sheet (0 for a zero-kerf cut).
        """
        self.sheet = sheet
        self.halfKerf = max(float(halfKerf), 0.0)
        self.points = numpy_support.vtk_to_numpy(sheet.GetPoints().GetData()).astype(float)
        self.triangles = numpy_support.vtk_to_numpy(sheet.GetPolys().GetConnectivityArray()).reshape(-1, 3)
        corners = self.points[self.triangles]
        cellNormals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        self.cellNormals = cellNormals / np.linalg.norm(cellNormals, axis=1, keepdims=True)

        outer, inner = self.points[0::2], self.points[1::2]
        pairCount = len(outer)
        self.closed = len(self.triangles) == 2 * pairCount  # an open sheet has one quad fewer
        middle = (outer + inner) / 2.0
        following = np.roll(middle, -1, axis=0) if self.closed else middle[1:]
        quadLengths = np.linalg.norm(following - middle[:len(following)], axis=1)
        quadCount = len(quadLengths)
        self.quadStarts = np.concatenate([[0.0], np.cumsum(quadLengths)])  # s of each quad's start
        self.period = float(self.quadStarts[-1])
        self.heights = np.linalg.norm(inner - outer, axis=1)
        self.foldDirections = (inner - outer) / self.heights[:, np.newaxis]  # along each fold line

        # Per triangle: its quad, and (s, v) of its vertices (the closing quad of a closed
        # sheet has its second pair at s = period, not 0)
        pairs = self.triangles // 2
        quads = pairs.min(axis=1)
        if self.closed:
            wraps = (pairs.max(axis=1) == pairCount - 1) & (quads == 0)
            quads[wraps] = pairCount - 1
        self.triangleQuads = quads
        s = np.where(pairs == quads[:, np.newaxis], self.quadStarts[quads][:, np.newaxis],
                     self.quadStarts[quads + 1][:, np.newaxis])
        v = (self.triangles % 2) * self.heights[pairs]
        self.triangleUV = np.stack([s, v], axis=2)  # (M, 3, 2)
        self.quadTriangles = np.full((quadCount, 2), -1, dtype=np.int64)
        for triangleId, quad in enumerate(quads):
            self.quadTriangles[quad, 0 if self.quadTriangles[quad, 0] < 0 else 1] = triangleId
        quadNormals = self.cellNormals[self.quadTriangles].sum(axis=1)
        self.quadNormals = quadNormals / np.linalg.norm(quadNormals, axis=1, keepdims=True)

        # Per quad: the direction in the sheet, across its inner edge and away from the sheet
        # (where the floor of a groove starts)
        innerTriangles = self.quadTriangles[np.arange(quadCount),
                                            np.argmax((self.triangles[self.quadTriangles] % 2).sum(axis=2), axis=1)]
        innerEdges = inner[(np.arange(quadCount) + 1) % pairCount] - inner[:quadCount]
        across = np.cross(innerEdges, self.cellNormals[innerTriangles])
        across *= np.sign(np.sum(across * self.foldDirections[:quadCount], axis=1))[:, np.newaxis]
        self.innerDirections = across / np.linalg.norm(across, axis=1, keepdims=True)

        # Folds: pair p joins quad p - 1 (before) and quad p (after)
        self.foldAngles = np.zeros(pairCount)
        self.foldTurns = np.zeros(pairCount)  # +1 if the sheet turns towards its +normal side
        foldPairs = range(pairCount) if self.closed else range(1, pairCount - 1)
        for p in foldPairs:
            before, after = self.quadNormals[(p - 1) % quadCount], self.quadNormals[p % quadCount]
            self.foldAngles[p] = np.arccos(np.clip(np.dot(before, after), -1.0, 1.0))
            direction = middle[(p + 1) % pairCount] - middle[p]
            self.foldTurns[p] = np.sign(np.dot(before, direction))
        # Never trim more than half a quad on either side of a crease
        before = quadLengths[(np.arange(pairCount) - 1) % quadCount]
        after = quadLengths[np.minimum(np.arange(pairCount), quadCount - 1)]
        self.maxCreaseTrims = 0.5 * np.minimum(before, after)

        self.locator = vtk.vtkCellLocator()
        self.locator.SetDataSet(sheet)
        self.locator.BuildLocator()

    def _layout(self, offsets: np.ndarray) -> dict[str, np.ndarray]:
        """How u relates to s on the level surfaces at the given signed offsets from the sheet.

        Where the sheet turns away from the offset side (convex), the level surface has a
        rounded corner of length |offset| * fold angle. Where it turns towards it (concave), the
        offset planes of the two faces meet in a crease |offset| * tan(fold angle / 2) before
        the fold, and the faces are trimmed there.

        :param offsets: (K,) signed offsets (> 0 on the + side).
        :return: per offset: "shifts" (K, Q) of each quad's u, "arcs" (K, P) arc length at each
            pair (0 if none), "arcStarts" (K, P) u where it starts, "trims" (K, P) crease trim
            at each pair (0 if none), "periods" (K,) period of u.
        """
        unique, inverse = np.unique(np.asarray(offsets, dtype=float), return_inverse=True)
        radii = np.abs(unique)[:, np.newaxis]
        folded = self.foldAngles > 0
        convex = folded & (self.foldTurns * np.sign(unique)[:, np.newaxis] < 0)
        concave = folded & ~convex
        arcs = np.where(convex, radii * self.foldAngles, 0.0)
        trims = np.where(concave, np.minimum(radii * np.tan(self.foldAngles / 2.0), self.maxCreaseTrims), 0.0)
        cumulative = np.cumsum(arcs - 2.0 * trims, axis=1)  # the fold at pair p lies just before quad p
        pairCount = len(self.foldAngles)
        layout = {"shifts": cumulative[:, :len(self.quadTriangles)], "arcs": arcs, "trims": trims,
                  "arcStarts": self.quadStarts[:pairCount] + cumulative - arcs + 2.0 * trims,
                  "periods": self.period + cumulative[:, -1]}
        inverse = inverse.ravel()
        return {name: values[inverse] for name, values in layout.items()}

    def _latitudes(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(theta, t) of chart coordinates w: the angle around the inner edge and the distance
        from it along the sheet."""
        r = self.halfKerf
        if not r > 0:
            return np.zeros(len(w)), np.maximum(-w, 0.0)
        theta = np.clip(w / r, 0.0, np.pi)
        t = np.where(w < 0.0, -w, np.maximum(w - r * np.pi, 0.0))
        return theta, t

    def _offsets(self, theta: np.ndarray) -> np.ndarray:
        """Signed offset from the sheet of the level at latitude theta (exactly +/- r on the sides)."""
        offsets = self.halfKerf * np.cos(theta)
        offsets[theta <= 0.0] = self.halfKerf
        offsets[theta >= np.pi] = -self.halfKerf
        return offsets

    def _innerHeights(self, quads: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Height H of the sheet (v of its inner edge) at s in the given quads."""
        pairCount = len(self.heights)
        starts, ends = self.quadStarts[quads], self.quadStarts[quads + 1]
        fraction = np.clip((s - starts) / np.maximum(ends - starts, 1e-300), 0.0, 1.0)
        return self.heights[quads] * (1.0 - fraction) + self.heights[(quads + 1) % pairCount] * fraction

    def periods(self, w: np.ndarray) -> np.ndarray:
        """Period of u at chart coordinates w (closed sheets)."""
        return self._layout(self._offsets(self._latitudes(np.asarray(w, dtype=float))[0]))["periods"]

    def signedDistances(self, w: np.ndarray) -> np.ndarray:
        """Signed distance to the sheet of cut surface points at chart coordinates w."""
        return np.where(np.asarray(w) <= self.halfKerf * np.pi / 2.0, self.halfKerf, -self.halfKerf)

    def chartCoordinates(self, points: np.ndarray) -> np.ndarray:
        """Chart coordinates (u, w) of points on the cut surface.

        :param points: (K, 3) points on the cut surface (the sheet if halfKerf = 0).
        :return: (K, 2) coordinates.
        """
        closest = np.empty_like(points, dtype=float)
        cellIds = np.empty(len(points), dtype=np.int64)
        position = [0.0, 0.0, 0.0]
        cellId, subId, distance2 = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
        for i, point in enumerate(points):
            self.locator.FindClosestPoint(point.tolist(), position, cellId, subId, distance2)
            closest[i], cellIds[i] = position, int(cellId)
        weights = self._barycentric3D(closest, self.points[self.triangles[cellIds]])
        s, v = np.einsum("kj,kjc->kc", weights, self.triangleUV[cellIds]).T
        quads = self.triangleQuads[cellIds]
        t = np.maximum(self._innerHeights(quads, s) - v, 0.0)
        r = self.halfKerf
        if not r > 0:
            return np.column_stack([s, -t])

        offsets = points - closest
        theta = np.arctan2(np.maximum(np.sum(offsets * self.innerDirections[quads], axis=1), 0.0),
                           np.sum(offsets * self.cellNormals[cellIds], axis=1))
        u = s + self._layout(self._offsets(theta))["shifts"][np.arange(len(s)), quads]

        # Points on a rounded corner: their closest sheet point is on the fold line (or at its
        # inner end, on the floor), and the offset lies in the wedge between the face normals
        pairCount = len(self.foldAngles)
        quadCount = len(self.quadTriangles)
        for pairs, onFold in (((quads + 1) % pairCount, self.quadStarts[quads + 1] - s < self.FOLD_TOLERANCE),
                              (quads, s - self.quadStarts[quads] < self.FOLD_TOLERANCE)):
            ids = np.flatnonzero(onFold & (self.foldAngles[pairs] > 0))
            if len(ids) == 0:
                continue
            p = pairs[ids]
            before, after = self.quadNormals[(p - 1) % quadCount], self.quadNormals[p % quadCount]
            along = np.sum(offsets[ids] * self.foldDirections[p], axis=1)
            perpendicular = offsets[ids] - along[:, np.newaxis] * self.foldDirections[p]
            radii = np.linalg.norm(perpendicular, axis=1)
            side = np.where(np.sum(perpendicular * (before + after), axis=1) >= 0, 1.0, -1.0)
            convex = (self.foldTurns[p] * side < 0) & (radii > 1e-12)
            ids, p, before, along, perpendicular, radii, side = (
                values[convex] for values in (ids, p, before, along, perpendicular, radii, side))
            if len(ids) == 0:
                continue
            theta[ids] = np.arctan2(np.maximum(along, 0.0), side * radii)
            layout = self._layout(self._offsets(theta[ids]))
            rows = np.arange(len(ids))
            angles = np.clip(np.arccos(np.clip(np.sum(perpendicular / radii[:, np.newaxis] * side[:, np.newaxis]
                                                      * before, axis=1), -1.0, 1.0)), 0.0, self.foldAngles[p])
            u[ids] = layout["arcStarts"][rows, p] + r * np.abs(np.cos(theta[ids])) * angles
        w = r * theta + np.where(theta < np.pi / 2.0, -t, t)
        return np.column_stack([u, w])

    def surfacePoints(self, coordinates: np.ndarray) -> np.ndarray:
        """Points on the cut surface at the given chart coordinates (inverse of chartCoordinates).

        :param coordinates: (K, 2) coordinates (u, w); u wraps for a closed sheet and is
            clamped for an open one.
        :return: (K, 3) points.
        """
        quadCount = len(self.quadTriangles)
        pairCount = len(self.foldAngles)
        r = self.halfKerf
        u, w = coordinates[:, 0].astype(float), coordinates[:, 1].astype(float)
        theta, t = self._latitudes(w)
        offsets = self._offsets(theta) if r > 0 else np.zeros(len(u))
        layout = self._layout(offsets)
        if self.closed:
            u = np.mod(u, layout["periods"])
        rows = np.arange(len(u))

        # Each quad's stretch of u, less the parts beyond a crease
        trims = layout["trims"]
        trimStarts = trims[:, :quadCount]
        trimEnds = trims[:, (np.arange(quadCount) + 1) % pairCount]
        flatStarts = self.quadStarts[:quadCount] + layout["shifts"] + trimStarts
        quads = np.clip(np.count_nonzero(flatStarts <= u[:, np.newaxis], axis=1) - 1, 0, quadCount - 1)
        s = np.clip(u - layout["shifts"][rows, quads], self.quadStarts[quads] + trimStarts[rows, quads],
                    self.quadStarts[quads + 1] - trimEnds[rows, quads])
        heights = self._innerHeights(quads, s)
        v = np.clip(heights - t, 0.0, heights)

        query = np.column_stack([s, v])
        best = np.full(len(u), -1, dtype=np.int64)
        bestWeights = np.zeros((len(u), 3))
        bestScore = np.full(len(u), -np.inf)
        for column in range(2):
            triangleIds = self.quadTriangles[quads, column]
            weights = self._barycentric2D(query, self.triangleUV[triangleIds])
            score = weights.min(axis=1)  # >= 0 inside the triangle
            better = score > bestScore
            best[better], bestWeights[better], bestScore[better] = triangleIds[better], weights[better], score[better]
        result = np.einsum("kj,kjc->kc", bestWeights, self.points[self.triangles[best]])
        if not r > 0:
            return result
        result += r * (np.cos(theta)[:, np.newaxis] * self.cellNormals[best]
                       + np.sin(theta)[:, np.newaxis] * self.innerDirections[quads])

        # Points on the rounded corners
        arcs, arcStarts = layout["arcs"], layout["arcStarts"]
        onArc = (arcs > 0) & (u[:, np.newaxis] >= arcStarts) & (u[:, np.newaxis] <= arcStarts + arcs)
        ids = np.flatnonzero(onArc.any(axis=1))
        if len(ids) > 0:
            p = np.argmax(onArc[ids], axis=1)
            side = np.sign(offsets[ids])[:, np.newaxis]
            radii = np.abs(offsets[ids])
            before = side * self.quadNormals[(p - 1) % quadCount]
            after = side * self.quadNormals[p % quadCount]
            towards = after - np.sum(after * before, axis=1, keepdims=True) * before
            towards /= np.linalg.norm(towards, axis=1, keepdims=True)
            angles = ((u[ids] - arcStarts[ids, p]) / radii)[:, np.newaxis]
            outer, inner = self.points[2 * p], self.points[2 * p + 1]
            foldPoints = outer + (np.clip(self.heights[p] - t[ids], 0.0, self.heights[p]) / self.heights[p])[:, np.newaxis] * (inner - outer)
            result[ids] = (foldPoints + radii[:, np.newaxis] * (np.cos(angles) * before + np.sin(angles) * towards)
                           + r * np.sin(theta[ids])[:, np.newaxis] * self.foldDirections[p])
        return result

    def bendPoints(self, w: np.ndarray, tolerance: float) -> np.ndarray:
        """Chart points along the bends of the cut surface, for placing cap points there.

        These are the folds of the sheet (zero kerf), the creases of the offset surfaces, and
        lines across their rounded corners spaced so that the chords stay within tolerance.

        :param w: (L,) chart coordinates w at which to place points along each bend.
        :param tolerance: allowed distance (mm) of a chord across a rounded corner from it.
        :return: (M, 2) chart coordinates.
        """
        folded = np.flatnonzero(self.foldAngles > 0)
        w = np.asarray(w, dtype=float)
        if len(folded) == 0 or len(w) == 0:
            return np.zeros((0, 2))
        quadCount = len(self.quadTriangles)
        theta, _t = self._latitudes(w)
        offsets = self._offsets(theta) if self.halfKerf > 0 else np.zeros(len(w))
        layout = self._layout(offsets)
        result = []
        for row, level in enumerate(w):
            radius = abs(offsets[row])
            for p in folded:
                arc = layout["arcs"][row, p]
                if arc > 0:
                    chord = np.sqrt(8.0 * radius * tolerance)  # sagitta of a chord on the radius
                    steps = max(1, int(np.ceil(arc / chord)))
                    values = layout["arcStarts"][row, p] + arc * np.arange(steps + 1) / steps
                else:
                    values = [self.quadStarts[p] + layout["shifts"][row, p % quadCount] + layout["trims"][row, p]]
                result.extend((value, level) for value in values)
        return np.array(result, dtype=float).reshape(-1, 2)

    def floorLevels(self, tolerance: float) -> np.ndarray:
        """Chart coordinates w of lines across the rounded floor (around the inner edge), spaced
        so that chords between them stay within tolerance; empty for a zero-kerf cut."""
        r = self.halfKerf
        if not r > 0:
            return np.zeros(0)
        steps = max(2, int(np.ceil(np.pi * r / np.sqrt(8.0 * r * tolerance))))
        return r * np.pi * np.arange(1, steps) / steps

    @staticmethod
    def _barycentric3D(points: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Barycentric weights (K, 3) of points lying in triangles given by corners (K, 3, 3)."""
        e0, e1, e2 = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], points - corners[:, 0]
        d00, d01, d11 = np.sum(e0 * e0, axis=1), np.sum(e0 * e1, axis=1), np.sum(e1 * e1, axis=1)
        d20, d21 = np.sum(e2 * e0, axis=1), np.sum(e2 * e1, axis=1)
        denominator = d00 * d11 - d01 * d01
        w1 = (d11 * d20 - d01 * d21) / denominator
        w2 = (d00 * d21 - d01 * d20) / denominator
        return np.column_stack([1.0 - w1 - w2, w1, w2])

    @staticmethod
    def _barycentric2D(points: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Barycentric weights (K, 3) of 2D points in triangles given by corners (K, 3, 2)."""
        e0, e1, e2 = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], points - corners[:, 0]
        denominator = e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]
        w1 = (e2[:, 0] * e1[:, 1] - e2[:, 1] * e1[:, 0]) / denominator
        w2 = (e0[:, 0] * e2[:, 1] - e0[:, 1] * e2[:, 0]) / denominator
        return np.column_stack([1.0 - w1 - w2, w1, w2])


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
    outlineModel - Model node showing where the cutting sheet meets the model surface.
    treatBoneAsSolid - Cut a solid version of the model (makeSolidPolyData), unless it is solid already.
    solidVoxelSize - Detail (voxel size, mm) of solid bone models.
    solidGapSeal - Holes and gaps up to about twice this (mm) are sealed in solid bone models.
    minSegmentPercent - Smaller pieces are joined to a neighbouring bone segment after a cut.
    clearanceMarkups - Points marking where the cut comes closest to each structure to protect.
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
    outlineModel: vtkMRMLModelNode
    treatBoneAsSolid: bool = True
    solidVoxelSize: Annotated[float, WithinRange(0.05, 5.0)] = 0.25
    solidGapSeal: Annotated[float, WithinRange(0.0, 10.0)] = 1.5
    # Pieces with less surface area than this percentage of the largest bone segment are joined
    # to the segment they touch most (joinSmallSegments); 0 keeps every piece
    minSegmentPercent: Annotated[float, WithinRange(0.0, 50.0)] = 1.0
    clearanceMarkups: vtkMRMLMarkupsFiducialNode


#
# OsteotomyCutsWidget
#


class OsteotomyCutsWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    # Delay after the last point edit before the sheet preview is rebuilt, and before the
    # distances to structures to protect are checked again (live preview)
    PREVIEW_DELAY_MS = 80
    CLEARANCE_DELAY_MS = 1000
    # A live distance check slower than this (s) turns live checking off for the session
    MAX_LIVE_CLEARANCE_SECONDS = 3.0

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
        self._currentLineId = None  # cut path whose settings the GUI shows (saved to it on edits)
        self._clearanceTimer = None  # throttles the live distance check to structures to protect
        self._clearanceResults = {}  # structure node ID -> last ClearanceResult (display only)
        self._structuresSignature = None  # structures shown in the table (rebuilt when it changes)
        self._liveClearance = True  # turned off for the session if the live check is slow

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
        self.ui.createSolidButton.connect("clicked(bool)", self.onCreateSolidButton)
        if importNdimage() is None:
            for widget in (self.ui.createSolidButton, self.ui.treatBoneAsSolidCheckBox):
                widget.toolTip = _("Solid bone models need scipy, which is missing from this Slicer installation.")
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.undoButton.connect("clicked(bool)", self.onUndoButton)
        self.ui.mergeButton.connect("clicked(bool)", self.onMergeButton)
        self.ui.mergeFragmentsSelector.connect("checkedNodesChanged()", self._updateMergeButton)
        self.ui.groupLinesSelector.connect("checkedNodesChanged()", self.onGroupLinesChanged)
        self.ui.sheetOpacitySliderWidget.connect("valueChanged(double)", self._applySheetOpacity)

        self._previewTimer = qt.QTimer()
        self._previewTimer.setSingleShot(True)
        self._previewTimer.setInterval(self.PREVIEW_DELAY_MS)
        self._previewTimer.connect("timeout()", self._updatePreview)

        self.ui.addStructureButton.connect("clicked(bool)", self.onAddStructure)
        self.ui.removeStructureButton.connect("clicked(bool)", self.onRemoveStructure)
        self.ui.checkDistancesButton.connect("clicked(bool)", self.onCheckDistances)
        self.ui.structuresTable.connect("itemSelectionChanged()", self._updateStructureEditor)
        self.ui.structureCategoryComboBox.connect("activated(int)", self._onStructureCategoryChosen)
        self.ui.structureSafeDistanceSpinBox.connect("valueChanged(double)", self._onStructureEditorChanged)
        self.ui.structureRadiusSpinBox.connect("valueChanged(double)", self._onStructureEditorChanged)
        self.ui.structureEnabledCheckBox.connect("toggled(bool)", self._onStructureEditorChanged)
        self.ui.structuresTable.horizontalHeader().setSectionResizeMode(qt.QHeaderView.ResizeToContents)
        self._clearanceTimer = qt.QTimer()
        self._clearanceTimer.setSingleShot(True)
        self._clearanceTimer.setInterval(self.CLEARANCE_DELAY_MS)
        self._clearanceTimer.connect("timeout()", lambda: self._runClearanceCheck(buildSolid=False, live=True))

        # Make sure parameter node is initialised (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        if self._previewTimer:
            self._previewTimer.stop()
        if self._clearanceTimer:
            self._clearanceTimer.stop()
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
            # exit() may have removed the observer already.
            if self.hasObserver(self._parameterNode.parameterNode, vtk.vtkCommand.ModifiedEvent,
                                self._updateGuiFromParameterNode):
                self.removeObserver(self._parameterNode.parameterNode, vtk.vtkCommand.ModifiedEvent,
                                    self._updateGuiFromParameterNode)
        self._observeMarkupsNodes([])
        self._parameterNode = inputParameterNode
        self._currentLineId = None
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
        self._syncLineSettings()

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

        self._observeMarkupsNodes(self.logic.getOsteotomyLines(self._parameterNode.cutCurve)
                                  + [self._parameterNode.directionLine])
        self._refreshStructuresTable()
        self._updateActionState()
        self._schedulePreview()

    def _syncLineSettings(self) -> None:
        """Each cut path keeps its own direction and saw settings: show them when another path
        is selected, and save edits to the selected one."""
        curveNode = self._parameterNode.cutCurve
        curveId = curveNode.GetID() if curveNode is not None else None
        if curveId != self._currentLineId:
            self._currentLineId = curveId
            if curveNode is not None:
                self.logic.loadLineSettings(self._parameterNode)  # re-enters with the same path
            self._updateGroupSelector()
        elif curveNode is not None:
            self.logic.storeLineSettings(self._parameterNode)

    def _updateGroupSelector(self) -> None:
        """Tick the further lines of the selected path's osteotomy and show the order."""
        curveNode = self._parameterNode.cutCurve if self._parameterNode else None
        firstLine = self.logic.findFirstLine(curveNode)
        lines = self.logic.getOsteotomyLines(curveNode)
        groupIds = {line.GetID() for line in lines[1:]}
        combo = self.ui.groupLinesSelector
        wasBlocked = combo.blockSignals(True)
        for index in range(combo.nodeCount()):
            node = combo.nodeFromIndex(index)
            combo.setCheckState(node, qt.Qt.Checked if node.GetID() in groupIds else qt.Qt.Unchecked)
        combo.blockSignals(wasBlocked)
        combo.enabled = firstLine is not None
        if firstLine is None:
            text = ""
        elif len(lines) == 1:
            text = _("Single line. Tick further lines to cut them together with this one in one step.")
        else:
            order = ", ".join(f"{index + 1}. {line.GetName()}" for index, line in enumerate(lines))
            text = _("Cut together, in this order: {order}. Each later line stops where it meets an earlier "
                     "one. Direction and saw settings below belong to the selected line, {line}.").format(
                order=order, line=curveNode.GetName())
        self.ui.groupOrderLabel.text = text

    def onGroupLinesChanged(self) -> None:
        """Store the ticked further lines of the osteotomy, keeping the order they were ticked in."""
        parameterNode = self._parameterNode
        if not parameterNode or parameterNode.cutCurve is None:
            return
        firstLine = self.logic.findFirstLine(parameterNode.cutCurve)
        checked = [node for node in self.ui.groupLinesSelector.checkedNodes() if node.GetID() != firstLine.GetID()]
        checkedIds = {node.GetID() for node in checked}
        current = self.logic.getGroupLines(firstLine)
        currentIds = {node.GetID() for node in current}
        ordered = ([node for node in current if node.GetID() in checkedIds]
                   + [node for node in checked if node.GetID() not in currentIds])
        self.logic.setGroupLines(firstLine, ordered)
        self._updateGroupSelector()
        self._updateGuiFromParameterNode()

    def _onMarkupsModified(self, caller=None, event=None) -> None:
        """Points of the cut path or direction line changed."""
        self._updateActionState()
        self._schedulePreview()

    def _onCurvePointsPlaced(self, caller=None, event=None) -> None:
        """After a point is placed or dragged, put the cut path back on the model surface."""
        parameterNode = self._parameterNode
        if (self._snapping or not parameterNode or not parameterNode.snapToSurface
                or parameterNode.inputModel is None or parameterNode.cutCurve is None or caller is None
                or caller.GetID() not in {line.GetID() for line in self.logic.getOsteotomyLines(parameterNode.cutCurve)}):
            return
        self._snapping = True
        try:
            self.logic.snapCurveToSurface(caller, parameterNode.inputModel)
        except ValueError:
            pass  # empty model or non-linear transform: validateInputs already reports it
        finally:
            self._snapping = False

    def _schedulePreview(self) -> None:
        """Rebuild the sheet preview shortly, once point edits pause (and check the distances to
        structures to protect a little later, if there are any)."""
        if self._previewTimer:
            self._previewTimer.start()
        if (self._clearanceTimer and self._liveClearance and self._parameterNode
                and self._parameterNode.livePreview and self.logic.getProtectedStructures()):
            self._clearanceTimer.start()

    #
    # Structures to protect
    #

    CATEGORY_ORDER = (StructureCategory.NERVE, StructureCategory.TOOTH, StructureCategory.OTHER)

    @staticmethod
    def _categoryText(category: StructureCategory) -> str:
        return {StructureCategory.NERVE: _("Nerve"), StructureCategory.TOOTH: _("Tooth root"),
                StructureCategory.OTHER: _("Other")}[category]

    @staticmethod
    def _statusText(result: ClearanceResult) -> str:
        if not np.isfinite(result.clearance):
            return _("Safe (the cut does not reach the bone)")
        return {ClearanceStatus.SAFE: _("Safe"), ClearanceStatus.TOO_CLOSE: _("Too close"),
                ClearanceStatus.ENTERS: _("Cut enters structure")}[result.status]

    def _selectedStructure(self) -> Optional[ProtectedStructure]:
        rows = self.ui.structuresTable.selectionModel().selectedRows()
        if not rows:
            return None
        item = self.ui.structuresTable.item(rows[0].row(), 0)
        node = slicer.mrmlScene.GetNodeByID(item.data(qt.Qt.UserRole)) if item is not None else None
        return self.logic.getProtectedStructure(node)

    def _refreshStructuresTable(self, force: bool = False) -> None:
        """Show the structures to protect with their last distances (rebuilt only when they change)."""
        structures = self.logic.getProtectedStructures()
        signature = tuple((s.node.GetID(), s.node.GetName(), s.node.GetAttribute(STRUCTURE_ATTRIBUTE))
                          for s in structures) + tuple(sorted(
            (nodeId, round(r.clearance, 2) if np.isfinite(r.clearance) else None, r.status.value)
            for nodeId, r in self._clearanceResults.items()))
        if signature == self._structuresSignature and not force:
            return
        self._structuresSignature = signature
        selected = self._selectedStructure()
        selectedId = selected.node.GetID() if selected is not None else None
        table = self.ui.structuresTable
        wasBlocked = table.blockSignals(True)
        table.setRowCount(len(structures))
        for row, structure in enumerate(structures):
            result = self._clearanceResults.get(structure.node.GetID())
            if not structure.enabled:
                distance, status = "", _("Not checked")
            elif result is None:
                distance, status = "", ""
            else:
                distance = f"{result.clearance:.1f} mm" if np.isfinite(result.clearance) else "-"
                status = self._statusText(result)
            texts = (structure.node.GetName(), self._categoryText(structure.category),
                     f"{structure.safeDistance:.1f} mm", distance, status)
            for column, text in enumerate(texts):
                item = qt.QTableWidgetItem(text)
                if column == 0:
                    item.setData(qt.Qt.UserRole, structure.node.GetID())
                if result is not None and structure.enabled and column >= 3:
                    item.setForeground(qt.QBrush(qt.QColor.fromRgbF(*self.logic.STATUS_COLOURS[result.status])))
                table.setItem(row, column, item)
            if structure.node.GetID() == selectedId:
                table.selectRow(row)
        table.blockSignals(wasBlocked)
        self._updateStructureEditor()

    def _updateStructureEditor(self) -> None:
        """Show the settings of the selected structure in the editor."""
        structure = self._selectedStructure()
        editors = (self.ui.structureCategoryComboBox, self.ui.structureSafeDistanceSpinBox,
                   self.ui.structureRadiusSpinBox, self.ui.structureEnabledCheckBox, self.ui.removeStructureButton)
        for editor in editors:
            editor.enabled = structure is not None
        if structure is None:
            return
        for editor in editors[:4]:
            editor.blockSignals(True)
        self.ui.structureCategoryComboBox.currentIndex = self.CATEGORY_ORDER.index(structure.category)
        self.ui.structureSafeDistanceSpinBox.value = structure.safeDistance
        self.ui.structureRadiusSpinBox.value = structure.radius
        self.ui.structureRadiusSpinBox.enabled = structure.node.IsA("vtkMRMLMarkupsCurveNode")
        self.ui.structureEnabledCheckBox.checked = structure.enabled
        for editor in editors[:4]:
            editor.blockSignals(False)

    def _onStructureCategoryChosen(self, index: int) -> None:
        """A chosen type also sets its usual safe distance."""
        category = self.CATEGORY_ORDER[index]
        self.ui.structureSafeDistanceSpinBox.blockSignals(True)
        self.ui.structureSafeDistanceSpinBox.value = DEFAULT_SAFE_DISTANCES[category]
        self.ui.structureSafeDistanceSpinBox.blockSignals(False)
        self._onStructureEditorChanged()

    def _onStructureEditorChanged(self, *args) -> None:
        """Store the editor's settings on the selected structure."""
        structure = self._selectedStructure()
        if structure is None:
            return
        structure.category = self.CATEGORY_ORDER[self.ui.structureCategoryComboBox.currentIndex]
        structure.safeDistance = self.ui.structureSafeDistanceSpinBox.value
        structure.radius = self.ui.structureRadiusSpinBox.value
        structure.enabled = self.ui.structureEnabledCheckBox.checked
        self.logic.setProtectedStructure(structure)
        self._clearanceResults.pop(structure.node.GetID(), None)
        self._refreshStructuresTable()
        self._schedulePreview()

    def onAddStructure(self) -> None:
        """Protect the model or curve chosen in the selector."""
        with slicer.util.tryWithErrorDisplay(_("Failed to add the structure."), waitCursor=True):
            node = self.ui.structureNodeSelector.currentNode()
            if node is None:
                raise ValueError(_("Choose a model or curve to protect first."))
            if self._parameterNode and self._parameterNode.inputModel is not None \
                    and node.GetID() == self._parameterNode.inputModel.GetID():
                raise ValueError(_("This is the bone being cut. Choose the nerve canal, teeth or another structure."))
            self.logic.addProtectedStructure(node)
            self._refreshStructuresTable(force=True)
            for row in range(self.ui.structuresTable.rowCount):
                if self.ui.structuresTable.item(row, 0).data(qt.Qt.UserRole) == node.GetID():
                    self.ui.structuresTable.selectRow(row)
            self._schedulePreview()

    def onRemoveStructure(self) -> None:
        """Stop protecting the selected structure."""
        structure = self._selectedStructure()
        if structure is not None:
            self.logic.removeProtectedStructure(structure.node)
            self._clearanceResults.pop(structure.node.GetID(), None)
            if not self.logic.getProtectedStructures() and self._parameterNode:
                self._clearanceResults = {}
                self.logic.updateClearanceDisplay(self._parameterNode, [])
                self.ui.structuresStatusLabel.text = ""
            self._refreshStructuresTable(force=True)
            self._schedulePreview()

    def onCheckDistances(self) -> None:
        """Measure the distances from the cut to the structures now (building the solid bone
        model if needed)."""
        self._runClearanceCheck(buildSolid=True, live=False)

    def _runClearanceCheck(self, buildSolid: bool, live: bool) -> Optional[list[ClearanceResult]]:
        """Check the distances to the structures to protect and show them (table, closest points,
        sheet colour). Live checks (preview) never build the solid model and are silent.

        :return: the results, or None if the check could not run.
        """
        parameterNode = self._parameterNode
        if not parameterNode or self.logic.validateInputs(parameterNode):
            return None
        start = time.perf_counter()
        results = None
        try:
            with slicer.util.tryWithErrorDisplay(_("Failed to check the distances."), waitCursor=True,
                                                 show=not live):
                results = self.logic.checkClearancesForParameters(parameterNode, buildSolid=buildSolid)
        except Exception:
            return None  # shown (or logged, when live) by tryWithErrorDisplay
        if results is None:
            return None
        elapsed = time.perf_counter() - start
        self._clearanceResults = {result.structure.node.GetID(): result for result in results}
        self.logic.updateClearanceDisplay(parameterNode, results)
        self._refreshStructuresTable()
        problems = [result for result in results if result.status != ClearanceStatus.SAFE]
        text = (_("Too close: {names}.").format(names=", ".join(r.structure.node.GetName() for r in problems))
                if problems else _("All checked structures are at a safe distance.") if results else "")
        if live and elapsed > self.MAX_LIVE_CLEARANCE_SECONDS:
            self._liveClearance = False
            text += " " + _("The distance check is slow on this model, so it is no longer repeated while points "
                            "move: press \"Check distances\" (it is always done before a cut).")
        self.ui.structuresStatusLabel.text = (text + " " + _(SAFETY_NOTE)).strip() if results else ""
        return results

    def _confirmClearances(self, parameterNode: OsteotomyCutsParameterNode) -> tuple[bool, list, bool]:
        """Before a cut, check the distances to the structures to protect and ask the surgeon to
        confirm if any is too close.

        :return: (go on, results, whether the surgeon overrode a warning).
        """
        if not [s for s in self.logic.getProtectedStructures() if s.enabled]:
            return True, [], False
        results = self._runClearanceCheck(buildSolid=True, live=False)
        if results is None:
            return False, [], False
        problems = [result for result in results if result.status != ClearanceStatus.SAFE]
        if not problems:
            return True, results, False
        lines = [_("{name} ({type}): {status}, {distance:.1f} mm from the cut (safe distance {safe:.1f} mm, "
                   "osteotomy line {line}).").format(
            name=r.structure.node.GetName(), type=self._categoryText(r.structure.category),
            status=self._statusText(r), distance=r.clearance, safe=r.structure.safeDistance, line=r.lineName)
            for r in problems]
        text = "\n".join(lines) + "\n\n" + _(SAFETY_NOTE)
        choice = self._askUser(_("Structures too close to the cut"), text, [_("Proceed anyway"), _("Cancel")],
                               default=1)
        return choice == 0, results, choice == 0

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
        haveScipy = importNdimage() is not None
        inputModel = self._parameterNode.inputModel
        self.ui.treatBoneAsSolidCheckBox.enabled = haveScipy
        self.ui.createSolidButton.enabled = (haveScipy and inputModel is not None
                                             and not self.logic.isSolidModel(inputModel))
        self.ui.applyButton.enabled = reason is None
        self.ui.applyButton.toolTip = reason or _("Cut the model along the cutting sheet.")
        self.ui.undoButton.enabled = bool(self.logic.getCurveResult(
            self.logic.findFirstLine(self._parameterNode.cutCurve)))
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

    @staticmethod
    def _isInteractive() -> bool:
        """False in tests and without a main window: dialogs are then logged, not shown."""
        return not slicer.app.testingEnabled() and slicer.util.mainWindow() is not None

    def _askUser(self, title: str, text: str, choices: list[str], default: int) -> int:
        """Ask the user to choose; the last choice is the one taken on Escape (Cancel).

        Headless or in tests, the message is logged and the default is returned.

        :return: index of the chosen button.
        """
        if not self._isInteractive():
            logging.info(f"OsteotomyCuts: {title}: {text} -> {choices[default]}")
            return default
        box = qt.QMessageBox(slicer.util.mainWindow())
        box.setWindowTitle(title)
        box.setIcon(qt.QMessageBox.Warning)
        box.setText(text)
        roles = [qt.QMessageBox.YesRole, qt.QMessageBox.AcceptRole, qt.QMessageBox.ApplyRole][:len(choices) - 1]
        roles.append(qt.QMessageBox.RejectRole)
        buttons = [box.addButton(label, role) for label, role in zip(choices, roles)]
        box.setDefaultButton(buttons[default])
        box.setEscapeButton(buttons[-1])
        box.exec_()
        clicked = box.clickedButton()
        role = box.buttonRole(clicked) if clicked is not None else qt.QMessageBox.RejectRole
        return roles.index(role) if role in roles else len(choices) - 1

    def _warnUser(self, title: str, text: str) -> None:
        """Show a warning (logged headless or in tests)."""
        if not self._isInteractive():
            logging.warning(f"OsteotomyCuts: {title}: {text}")
            return
        slicer.util.warningDisplay(text, windowTitle=title)

    def _confirmModelQuality(self, parameterNode: OsteotomyCutsParameterNode) -> bool:
        """Before a cut, check that the model to cut is closed and in one piece; if not, tell
        the surgeon and offer a solid bone model.

        :return: True to go on with the cut (possibly on a new solid model), False to stop.
        """
        quality = None
        with slicer.util.tryWithErrorDisplay(_("Failed to check the bone model."), waitCursor=True):
            quality = self.logic.assessModelToCut(parameterNode)
            if quality is None:
                return True  # made solid for the cut: closed
        if quality is None:
            return False  # the check failed (error shown)
        problems = []
        if not quality.isClosed:
            problems.append(_("its surface is not closed ({open} open edges, {crossing} edges where surfaces "
                              "cross), so the cut may not separate the bone and the bone segments may not be "
                              "closed").format(open=quality.openEdges, crossing=quality.nonManifoldEdges))
        if quality.enclosedShellCount:
            problems.append(_("it has {count} internal surfaces (e.g. marrow, canals, inner side of the cortex), "
                              "which are cut too").format(count=quality.enclosedShellCount))
        if quality.pieceCount - quality.enclosedShellCount > 1:
            problems.append(_("it is made of {count} separate pieces").format(
                count=quality.pieceCount - quality.enclosedShellCount))
        if not problems:
            return True
        name = parameterNode.inputModel.GetName()
        summary = _("The bone model {name} is not a clean solid: {problems}.").format(
            name=name, problems="; ".join(problems))
        self.ui.statusLabel.text = summary
        canMakeSolid = importNdimage() is not None
        choices = ([_("Create solid bone model")] if canMakeSolid else []) + [_("Proceed anyway"), _("Cancel")]
        text = summary + "\n\n" + (_("A solid bone model (holes sealed, internal surfaces filled) is recommended; "
                                     "the original model is kept.") if canMakeSolid else
                                   _("Solid bone models need scipy, which is missing."))
        choice = self._askUser(_("Check the bone model"), text, choices, default=len(choices) - 2)
        if canMakeSolid and choice == 0:
            self.onCreateSolidButton()
            return self.logic.isSolidModel(parameterNode.inputModel)
        return choice == len(choices) - 2

    def onApplyButton(self) -> None:
        """Check the model, cut it with a progress dialog, then check the bone segments."""
        parameterNode = self._requireParameterNode()
        if not self._confirmModelQuality(parameterNode):
            self._updateActionState()
            return
        proceed, clearances, override = self._confirmClearances(parameterNode)
        if not proceed:
            self._updateActionState()
            return
        progress = slicer.util.createProgressDialog(labelText=_("Cutting..."), maximum=100)

        def reportProgress(percent: int, message: str) -> None:
            progress.labelText = message
            progress.value = percent
            slicer.app.processEvents()

        notClosed = []
        try:
            with slicer.util.tryWithErrorDisplay(_("Failed to cut the model."), waitCursor=True):
                segments = self.logic.applyCut(parameterNode, reportProgress)
                if clearances:
                    self.logic.recordSafetyResults(segments, clearances, override)
                if len(segments) == 1:
                    message = _("The cut made a groove: the bone was not separated (1 bone segment).")
                else:
                    message = _("{count} bone segments created.").format(count=len(segments))
                if self.logic.lastJoinedPieces:
                    message += " " + _("{count} small pieces were joined to a neighbouring bone segment "
                                       "(Advanced).").format(count=self.logic.lastJoinedPieces)
                notClosed = [result.name for result in self.logic.lastSegmentQuality if not result.isClosed]
                if notClosed:
                    message += " " + _("WARNING: not closed: {names}.").format(names=", ".join(notClosed))
                else:
                    message += " " + _("All bone segments are closed.")
                self._resultMessage = message
        finally:
            progress.close()
            self._updateActionState()
        if notClosed:
            self._warnUser(_("Bone segments not closed"), _(
                "These bone segments are not closed: {names}.\n\nThey may not print correctly and their volumes "
                "are not reliable. Keep \"Cap cut faces\" on, and use \"Treat bone as solid\" or \"Create solid "
                "bone model\" for a model with holes.").format(names=", ".join(notClosed)))

    def onCreateSolidButton(self) -> None:
        """Create a solid version of the model to cut and select it, with a progress dialog."""
        parameterNode = self._requireParameterNode()
        progress = slicer.util.createProgressDialog(labelText=_("Making the bone solid..."), maximum=100)

        def reportProgress(percent: int, message: str) -> None:
            progress.labelText = message
            progress.value = percent
            slicer.app.processEvents()

        try:
            with slicer.util.tryWithErrorDisplay(_("Failed to create the solid bone model."), waitCursor=True):
                solidNode = self.logic.createSolidModel(parameterNode.inputModel, parameterNode.solidVoxelSize,
                                                        parameterNode.solidGapSeal, reportProgress)
                parameterNode.inputModel = solidNode
                self._resultMessage = _("Solid bone model {name} created.").format(name=solidNode.GetName())
        finally:
            progress.close()
            self._updateActionState()

    def onUndoButton(self) -> None:
        """Remove the fragments of the selected cut path and show its input model again."""
        with slicer.util.tryWithErrorDisplay(_("Failed to undo the cut."), waitCursor=True):
            self.logic.removeCutResult(self.logic.findFirstLine(self._requireParameterNode().cutCurve))
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
        # (cache key, vtkStaticCellLocator on the triangulated world mesh) of the last used model
        # surface, for snapping and the cut outline
        self._surfaceLocatorCache = None
        # Model node ID -> (cache key, solid world mesh), most recently used last
        self._solidCache = {}
        # Small pieces joined to a neighbouring bone segment by the last applyCut, and the check
        # of its bone segments (assessSegments)
        self.lastJoinedPieces = 0
        self.lastSegmentQuality = []

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

    # Kerf crossings along edges are located to this distance error (mm)
    ROOT_TOLERANCE = 1e-9
    MAX_ROOT_ITERATIONS = 50
    # Vertices this close (mm) to the kerf surface are taken to lie on it, so that crossings
    # next to them do not make slivers
    CUT_SNAP_DISTANCE = 1e-4

    # Points within this distance (mm, relative to 1 + kerf / 2) of the cut surface are on it
    CUT_SURFACE_TOLERANCE = 1e-6
    # Cap edges are split while their midpoint is farther than this (mm) from the cut surface
    # and they are longer than CAP_MIN_EDGE_LENGTH (mm)
    CAP_TOLERANCE = 0.01
    CAP_MIN_EDGE_LENGTH = 0.25
    # A cap is refined with at most max(CAP_MIN_POINT_BUDGET, CAP_POINT_BUDGET_PER_RIM_POINT x
    # its rim points) added points; normal caps need a fraction of that
    CAP_MIN_POINT_BUDGET = 50_000
    CAP_POINT_BUDGET_PER_RIM_POINT = 20
    # At most this many rim edges missing from a cap's Delaunay triangulation are recovered by
    # edge flips; with more, the slower fallback triangulation is used
    MAX_RECOVERED_EDGES = 500
    # Rounds of simultaneous Delaunay edge flips on a cap
    MAX_FLIP_ROUNDS = 1000
    # Largest spacing (mm) of cap points placed along a bend of the cut surface
    CAP_MAX_BEND_SPACING = 2.0
    # Points filling a cap are spaced at most this far apart (mm); the distance raster used to
    # place them has at most MAX_FILL_PIXELS pixels
    CAP_MAX_FILL_SPACING = 8.0
    MAX_FILL_PIXELS = 4_000_000
    # Rim edges shorter than this fraction of the model's bounding-box diagonal are collapsed
    RIM_COLLAPSE_FRACTION = 1e-4
    # ... and shorter than these fractions of half the kerf width and of the median rim edge
    RIM_COLLAPSE_KERF_FRACTION = 0.1
    RIM_COLLAPSE_EDGE_FRACTION = 0.1
    # Fragment normals are not smoothed across edges sharper than this (degrees), so that cut
    # faces are shaded flat
    NORMALS_FEATURE_ANGLE = 30.0
    # A limited cut (depth or end reach set) with the ideal blade (kerf 0) removes this thin
    # layer (mm): a zero-width cut follows the sheet's zero level, which continues past the
    # sheet's edges through the whole model, whereas the kerf ends where the sheet ends
    LIMITED_IDEAL_KERF_WIDTH = 0.1
    # Line width (pixels) of the cut outline preview
    OUTLINE_LINE_WIDTH = 4.0

    # Solid bone models (makeSolidPolyData): at most this many voxels in the working volume
    # (a few bytes each in several arrays)
    MAX_SOLID_VOXELS = 150_000_000
    # Solid pieces smaller than this fraction of the largest piece's volume are dropped (specks)
    SOLID_MIN_PIECE_FRACTION = 0.01
    # Distance transforms of the closing run on slabs of about this many voxels
    SOLID_SLAB_VOXELS = 8_000_000
    # Pass band of the windowed sinc smoothing of the voxel surface (as Slicer's segmentations)
    SOLID_SMOOTHING_PASS_BAND = 0.01
    # Solid copies of this many models are kept for sequential cuts
    SOLID_CACHE_SIZE = 3

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
                           depth: Optional[float | np.ndarray] = None,
                           endExtension: Optional[float] = None) -> vtk.vtkPolyData:
        """Build a ruled cutting sheet by extruding a polyline.

        The direction d points into the model (away from the viewer). Each path point P is
        extruded outward to A = P - extent * d and inward to B = P + depth * d (``depth``
        defaults to ``extent``, a through-cut). An open path is also extended by
        ``endExtension`` (default ``extent``, so that the sheet edges lie outside the model) at
        both ends, along the end tangent with its component along d removed.

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
        :param endExtension: distance (mm) the sheet continues past the ends of an open path.
            None means ``extent``.
        :return: triangulated sheet with consistent winding, point and cell normals.
        :raises ValueError: too few distinct points, a zero or malformed direction, a
            non-positive extent, depth or end extension, or a path segment (nearly) parallel to
            its direction.
        """
        points = np.asarray(pathPoints, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(_("Path points must be an (N, 3) array."))
        if not extent > 0:
            raise ValueError(_("Sheet extent must be positive."))
        endExtension = float(extent) if endExtension is None else float(endExtension)
        if not endExtension > 0:
            raise ValueError(_("The distance the cut continues past the line ends must be positive."))

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
            startPoint = points[0] + endExtension * self._endExtensionDirection(points[0] - points[1], dirs[0])
            endPoint = points[-1] + endExtension * self._endExtensionDirection(points[-1] - points[-2], dirs[-1])
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

        This is the zero-kerf cut: the mesh is clipped at signed distance 0. A cut with a kerf
        uses removeKerf instead.

        :param polyWithDistance: mesh with the distance point array (computeSheetDistance).
        :param options: cut options; must have kerfWidth = 0.
        :param arrayName: name of the distance array.
        :return: (positive side, negative side), both triangulated; either may be empty.
        :raises ValueError: for kerfWidth > 0 (use removeKerf).
        """
        if options.kerfWidth > 0:
            raise ValueError(_("A cut with a kerf width removes material; use removeKerf."))
        polyWithDistance.GetPointData().SetActiveScalars(arrayName)
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(polyWithDistance)
        clipper.SetValue(0.0)
        clipper.GenerateClippedOutputOn()
        clipper.Update()
        return (self._ensureTriangles(clipper.GetOutput()),
                self._ensureTriangles(clipper.GetClippedOutput()))

    def removeKerf(self, polyWithDistance: vtk.vtkPolyData, sheetPolyData: vtk.vtkPolyData, kerfWidth: float,
                   arrayName: str = "SheetDistance",
                   protectedName: Optional[str] = None) -> tuple[vtk.vtkPolyData, bool]:
        """Remove the material the saw blade takes away: everything closer than kerfWidth / 2 to
        the sheet (unsigned distance).

        The unsigned distance is to the sheet itself, so a depth-limited sheet removes a groove
        with a rounded bottom and no sign problems arise past the sheet's edges. Triangles are
        clipped where an edge crosses |d| = kerfWidth / 2; the crossing point is found on the
        exact distance along the edge (bracketed root finding), not by linear interpolation,
        which is off near folds of the sheet where |d| has a kink. Crossing points are shared by
        the triangles on both sides of an edge, so no cracks appear, and orientation is kept.
        The result is not separated into pieces (a groove leaves one piece) and the cut faces
        are left open.

        :param polyWithDistance: mesh with the signed distance point array (computeSheetDistance).
        :param sheetPolyData: the cutting sheet, to evaluate the distance along edges.
        :param kerfWidth: saw blade width (mm), > 0.
        :param arrayName: name of the signed distance array; it is kept in the output (exactly
            +/- kerfWidth / 2 at the crossing points).
        :param protectedName: optional integer point array, non-zero at points that are never
            removed (beyond an earlier line of the osteotomy that this line stops at). Such
            points may border removed ones only across an earlier kerf gap, which holds no mesh.
        :return: (remaining triangle mesh, whether any material was removed).
        :raises ValueError: for a non-positive kerf width, or StopConflictError if a removed
            point shares an edge with a protected point inside the kerf (the line meets bone
            across an earlier line outside that line's cut).
        """
        if not kerfWidth > 0:
            raise ValueError(_("The kerf width must be positive."))
        halfKerf = kerfWidth / 2.0
        mesh = self._ensureTriangles(polyWithDistance)
        arrays, normalsName, scalarsName = self._pointArrays(mesh)
        signed = arrays[arrayName][0].astype(float)
        # Vertices (almost) on the kerf surface are kept as they are and become rim points
        onSurface = np.abs(np.abs(signed) - halfKerf) <= self.CUT_SNAP_DISTANCE
        signed[onSurface] = np.where(signed[onSurface] >= 0, halfKerf, -halfKerf)
        arrays[arrayName] = (signed.astype(arrays[arrayName][0].dtype), arrays[arrayName][1])
        keep = np.abs(signed) >= halfKerf
        triangles = numpy_support.vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3).astype(np.int64)
        if protectedName is not None and protectedName in arrays:
            protected = arrays[protectedName][0] != 0
            blocked = protected & ~keep  # inside the kerf, but beyond an earlier line
            if np.any(blocked):
                removed = ~keep & ~protected
                for a, b in ((0, 1), (1, 2), (2, 0)):
                    if np.any((removed[triangles[:, a]] & blocked[triangles[:, b]])
                              | (blocked[triangles[:, a]] & removed[triangles[:, b]])):
                        raise StopConflictError()
            keep = keep | protected
        if np.all(keep):
            return mesh, False

        points = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData()).astype(float)
        pointCount = len(points)
        keptCorners = keep[triangles]
        keptCounts = keptCorners.sum(axis=1)

        # Edges with one kept and one removed end, each once
        partial = np.flatnonzero((keptCounts == 1) | (keptCounts == 2))
        edges = np.vstack([triangles[partial][:, [0, 1]], triangles[partial][:, [1, 2]], triangles[partial][:, [2, 0]]])
        edges = edges[keep[edges[:, 0]] != keep[edges[:, 1]]]
        edgeKeys = np.unique(np.minimum(edges[:, 0], edges[:, 1]) * pointCount + np.maximum(edges[:, 0], edges[:, 1]))
        first, second = edgeKeys // pointCount, edgeKeys % pointCount
        removedEnd = np.where(keep[first], second, first)
        keptEnd = np.where(keep[first], first, second)

        # Crossing along each edge from the removed end (t = 0) to the kept end (t = 1)
        implicitDistance = vtk.vtkImplicitPolyDataDistance()
        implicitDistance.SetInput(sheetPolyData)

        def excess(positions: np.ndarray) -> np.ndarray:
            values = vtk.vtkDoubleArray()
            implicitDistance.FunctionValue(numpy_support.numpy_to_vtk(positions, deep=True), values)
            return np.abs(numpy_support.vtk_to_numpy(values)) - halfKerf

        start, direction = points[removedEnd], points[keptEnd] - points[removedEnd]
        tLow, tHigh = np.zeros(len(edgeKeys)), np.ones(len(edgeKeys))
        fLow, fHigh = np.abs(signed[removedEnd]) - halfKerf, np.abs(signed[keptEnd]) - halfKerf
        atKeptEnd = fHigh <= 1e-12  # the kept end is on the cut surface itself
        t = np.where(atKeptEnd, 1.0, fLow / (fLow - fHigh))
        active = ~atKeptEnd
        for _iteration in range(self.MAX_ROOT_ITERATIONS):
            if not np.any(active):
                break
            ids = np.flatnonzero(active)
            f = excess(start[ids] + t[ids, np.newaxis] * direction[ids])
            converged = np.abs(f) <= self.ROOT_TOLERANCE
            below = f < 0
            # Illinois method: regula falsi, halving the value at an end that stays put
            tLow[ids[below]], fLow[ids[below]] = t[ids[below]], f[below]
            fHigh[ids[below]] *= 0.5
            tHigh[ids[~below]], fHigh[ids[~below]] = t[ids[~below]], f[~below]
            fLow[ids[~below]] *= 0.5
            active[ids[converged]] = False
            ids = ids[~converged]
            t[ids] = tLow[ids] - fLow[ids] * (tHigh[ids] - tLow[ids]) / (fHigh[ids] - fLow[ids])

        # New points and their point data
        newCount = int(np.count_nonzero(~atKeptEnd))
        crossingIds = np.where(atKeptEnd, keptEnd, 0)
        crossingIds[~atKeptEnd] = pointCount + np.arange(newCount)
        newT = t[~atKeptEnd][:, np.newaxis]
        newPoints = start[~atKeptEnd] + newT * direction[~atKeptEnd]
        for name, (values, dataType) in arrays.items():
            a, b = removedEnd[~atKeptEnd], keptEnd[~atKeptEnd]
            if name == arrayName:
                newValues = np.sign(signed[b]) * halfKerf
            elif np.issubdtype(values.dtype, np.floating):
                weight = newT if values.ndim == 2 else newT[:, 0]
                newValues = values[a] * (1.0 - weight) + values[b] * weight
                if name == normalsName:
                    norms = np.linalg.norm(newValues, axis=1, keepdims=True)
                    newValues = newValues / np.where(norms > 0, norms, 1.0)
            else:
                newValues = values[b]
            arrays[name] = (np.concatenate([values, newValues.astype(values.dtype)]), dataType)
        points = np.vstack([points, newPoints])

        def crossing(v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
            keys = np.minimum(v0, v1) * pointCount + np.maximum(v0, v1)
            return crossingIds[np.searchsorted(edgeKeys, keys)]

        def rotated(rows: np.ndarray, shift: np.ndarray) -> np.ndarray:
            order = (shift[:, np.newaxis] + np.arange(3)) % 3
            return np.take_along_axis(triangles[rows], order, axis=1)

        result = [triangles[keptCounts == 3]]
        rows = np.flatnonzero(keptCounts == 1)  # kept corner first: one smaller triangle
        v = rotated(rows, np.argmax(keptCorners[rows], axis=1))
        result.append(np.column_stack([v[:, 0], crossing(v[:, 0], v[:, 1]), crossing(v[:, 2], v[:, 0])]))
        rows = np.flatnonzero(keptCounts == 2)  # removed corner last: the remaining quad
        v = rotated(rows, (np.argmin(keptCorners[rows], axis=1) + 1) % 3)
        m12, m20 = crossing(v[:, 1], v[:, 2]), crossing(v[:, 2], v[:, 0])
        result += [np.column_stack([v[:, 0], v[:, 1], m12]), np.column_stack([v[:, 0], m12, m20])]
        triangles = np.vstack(result)
        triangles = triangles[(triangles[:, 0] != triangles[:, 1]) & (triangles[:, 1] != triangles[:, 2])
                              & (triangles[:, 2] != triangles[:, 0])]

        # Drop the removed points
        used = np.unique(triangles)
        newIds = np.full(len(points), -1, dtype=np.int64)
        newIds[used] = np.arange(len(used))
        arrays = {name: (values[used], dataType) for name, (values, dataType) in arrays.items()}
        return self._buildTriangleMesh(points[used], newIds[triangles], arrays, normalsName, scalarsName), True

    def capCutFaces(self, mesh: vtk.vtkPolyData, sheetMap: SheetParameterisation,
                    arrayName: str = "SheetDistance") -> tuple[vtk.vtkPolyData, int]:
        """Close the open cut faces a sheet left in a mesh, so that the pieces are watertight.

        The cut face boundary is the set of open edges whose ends lie on the cut surface
        (|d| = kerf / 2); rim edges too short to tell apart are collapsed first. The loops are
        mapped to the chart of the cut surface (SheetParameterisation), which unrolls both of
        its sides, the rounded floor of a groove, folds, creases and rounded corners into one
        plane, and triangulated there with vtkContourTriangulator (nested loops, e.g. an
        inferior alveolar canal, become holes). A groove (depth-limited cut) is thus lined
        with one cap along its walls and floor. The cap reuses the loop vertices, so it shares
        its edges with the mesh. Points are added on the cut surface along its bends and across
        a groove floor, the triangulation is made Delaunay by edge flips (no fans of slivers),
        and cap edges whose midpoint still strays more than CAP_TOLERANCE from the cut surface
        are split.

        :param mesh: triangle mesh with the signed distance array (after splitByDistance, one
            side only, or removeKerf).
        :param sheetMap: chart of the cut surface of the sheet that made the cut (its halfKerf
            is the offset of the cut surface).
        :param arrayName: name of the signed distance array.
        :return: (mesh with caps, 1 if the cut faces could not be capped, else 0).
        """
        triangles = numpy_support.vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3).astype(np.int64)
        points = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData()).astype(float)
        arrays, normalsName, scalarsName = self._pointArrays(mesh)
        distances = arrays[arrayName][0]
        halfKerf = sheetMap.halfKerf

        # Open edges, in the direction of the triangle that uses them
        openEdges = self._openEdges(triangles, len(points))
        tolerance = self.CUT_SURFACE_TOLERANCE * (1.0 + halfKerf)
        onCut = np.abs(np.abs(distances) - halfKerf) <= tolerance
        cutEdges = openEdges[onCut[openEdges[:, 0]] & onCut[openEdges[:, 1]]]
        if len(cutEdges) == 0:
            return mesh, 0

        # Collapse tiny rim edges (from crossings next to slivers): the triangulation of the
        # loops cannot tell such points apart. Kept well below the kerf and the typical rim edge,
        # so that a thin kerf's rounded floor and a finely refined rim are not collapsed.
        lengths = np.linalg.norm(points[cutEdges[:, 0]] - points[cutEdges[:, 1]], axis=1)
        diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        collapseLength = min(self.RIM_COLLAPSE_FRACTION * diagonal,
                             self.RIM_COLLAPSE_EDGE_FRACTION * float(np.median(lengths)))
        if halfKerf > 0:
            collapseLength = min(collapseLength, self.RIM_COLLAPSE_KERF_FRACTION * halfKerf)
        short = lengths < collapseLength
        if np.any(short):
            root = np.arange(len(points))

            def find(i: int) -> int:
                while root[i] != i:
                    root[i] = root[root[i]]
                    i = root[i]
                return i

            for a, b in cutEdges[short]:
                ra, rb = find(int(a)), find(int(b))
                if ra != rb:
                    root[max(ra, rb)] = min(ra, rb)
            while True:  # point every id straight at its root
                deeper = root[root]
                if np.array_equal(deeper, root):
                    break
                root = deeper
            triangles = root[triangles]
            triangles = triangles[(triangles[:, 0] != triangles[:, 1]) & (triangles[:, 1] != triangles[:, 2])
                                  & (triangles[:, 2] != triangles[:, 0])]
            openEdges = self._openEdges(triangles, len(points))
            cutEdges = openEdges[onCut[openEdges[:, 0]] & onCut[openEdges[:, 1]]]
            if len(cutEdges) == 0:
                return mesh, 1

        result = self._capLoops(cutEdges, points, sheetMap, arrays, arrayName)
        if result is None:
            return mesh, 1
        capPoints, capTriangles, capArrays = result
        for name in arrays:
            values, dataType = arrays[name]
            arrays[name] = (np.concatenate([values, capArrays[name].astype(values.dtype)]), dataType)
        return self._buildTriangleMesh(np.vstack([points, capPoints]), np.vstack([triangles, capTriangles]),
                                       arrays, normalsName, scalarsName), 0

    @staticmethod
    def _openEdges(triangles: np.ndarray, pointCount: int) -> np.ndarray:
        """Edges used by one triangle only, directed as in that triangle.

        Edges are compared as single integer keys: sorting rows (np.unique with axis=0) takes
        seconds on the millions of edges of a segmented bone.
        """
        directed = np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
        keys = np.minimum(directed[:, 0], directed[:, 1]) * pointCount + np.maximum(directed[:, 0], directed[:, 1])
        _unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        return directed[counts[inverse.ravel()] == 1]

    def _capLoops(self, edges: np.ndarray, points: np.ndarray, sheetMap: SheetParameterisation,
                  arrays: dict, arrayName: str) -> Optional[tuple[np.ndarray, np.ndarray, dict]]:
        """Triangulate the cut face bounded by the given open edges (one sheet).

        :param edges: (E, 2) open edges on the cut surface, directed as in the mesh.
        :param points: (N, 3) mesh points.
        :param sheetMap: chart of the cut surface.
        :param arrays: mesh point data {name: (values, VTK type)}.
        :param arrayName: name of the signed distance array.
        :return: (added points (K, 3), cap triangles in mesh ids (C, 3), with added points
            numbered from N, point data of added points {name: values}), or None if the loops
            could not be triangulated.
        """
        firstNewId = len(points)
        loopIds, localEdges = np.unique(edges, return_inverse=True)
        localEdges = localEdges.reshape(-1, 2)
        loopPoints = points[loopIds]
        chart = sheetMap.chartCoordinates(loopPoints)
        wMin, wMax = float(chart[:, 1].min()), float(chart[:, 1].max())
        # A closed sheet is unrolled to an annulus, so loops around it stay closed loops; the
        # innermost loop point lies where the annulus is true to length
        baseRadius = float(sheetMap.periods(chart[:, 1]).max()) / (2.0 * np.pi)

        def toPlane(coordinates: np.ndarray) -> np.ndarray:
            if not sheetMap.closed:
                return coordinates.copy()
            angle = 2.0 * np.pi * coordinates[:, 0] / sheetMap.periods(coordinates[:, 1])
            radius = baseRadius + coordinates[:, 1] - wMin
            return radius[:, np.newaxis] * np.column_stack([np.cos(angle), np.sin(angle)])

        def fromPlane(planar: np.ndarray) -> np.ndarray:
            if not sheetMap.closed:
                return planar
            w = np.linalg.norm(planar, axis=1) - baseRadius + wMin
            angle = np.mod(np.arctan2(planar[:, 1], planar[:, 0]), 2.0 * np.pi)
            return np.column_stack([angle / (2.0 * np.pi) * sheetMap.periods(w), w])

        def surfacePoints(planar: np.ndarray) -> np.ndarray:
            return sheetMap.surfacePoints(fromPlane(planar))

        def signedDistances(planar: np.ndarray) -> np.ndarray:
            return sheetMap.signedDistances(fromPlane(planar)[:, 1])

        planar = toPlane(chart)
        positions = loopPoints.copy()

        def edgeDeviation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
            """Distance of each edge's midpoint from the cut surface (current planar, positions)."""
            onSurface = surfacePoints((planar[first] + planar[second]) / 2.0)
            return np.linalg.norm(onSurface - (positions[first] + positions[second]) / 2.0, axis=1)

        localArrays = {name: values[loopIds] for name, (values, _dataType) in arrays.items()}
        loopCount = len(loopIds)

        # Points along the bends of the cut surface and across a groove floor, so that no
        # triangle spans a bend
        loopLengths = np.linalg.norm(positions[localEdges[:, 0]] - positions[localEdges[:, 1]], axis=1)
        spacing = float(np.clip(np.median(loopLengths), self.CAP_MIN_EDGE_LENGTH, self.CAP_MAX_BEND_SPACING))
        candidates = [sheetMap.bendPoints(np.arange(wMin + spacing / 2.0, wMax, spacing), self.CAP_TOLERANCE)]
        floorLevels = sheetMap.floorLevels(self.CAP_TOLERANCE)
        for level in floorLevels[(floorLevels > wMin) & (floorLevels < wMax)]:
            if sheetMap.closed:
                uValues = np.arange(0.0, float(sheetMap.periods(np.array([level]))[0]), spacing)
            else:
                uValues = np.arange(chart[:, 0].min() + spacing / 2.0, chart[:, 0].max(), spacing)
            candidates.append(np.column_stack([uValues, np.full(len(uValues), level)]))
        candidatePlanar = toPlane(np.vstack(candidates))
        loopStarts, loopEnds = planar[localEdges[:, 0]], planar[localEdges[:, 1]]
        if len(candidatePlanar) > 0:
            candidatePlanar = candidatePlanar[self._insideLoops(candidatePlanar, loopStarts, loopEnds)]
            clearance = self._distanceToSegments(candidatePlanar, loopStarts, loopEnds)
            candidatePlanar = candidatePlanar[clearance > spacing / 2.0]
        # Points filling the rest of the face, coarser away from the rim and the bends: a cap
        # of long slivers would be shredded by the refinement of a later cut through it
        candidatePlanar = np.vstack([candidatePlanar, self._gradedFillPoints(loopStarts, loopEnds,
                                                                             candidatePlanar, spacing)])
        # A loop edge whose diametral circle holds no other point is a Delaunay edge
        loopLengths2D = np.linalg.norm(loopEnds - loopStarts, axis=1)
        long = loopLengths2D > spacing  # shorter edges: the points keep spacing / 2 off them
        if np.any(long) and len(candidatePlanar) > 0:
            centres, radii = (loopStarts[long] + loopEnds[long]) / 2.0, 0.55 * loopLengths2D[long]
            clear = np.ones(len(candidatePlanar), dtype=bool)
            for chunk in range(0, len(candidatePlanar), 512):
                offsets = candidatePlanar[chunk:chunk + 512, np.newaxis, :] - centres[np.newaxis]
                clear[chunk:chunk + 512] = np.all(np.linalg.norm(offsets, axis=2) > radii, axis=1)
            candidatePlanar = candidatePlanar[clear]

        # Delaunay triangulation of all the points, kept inside the loops; if it does not keep
        # every loop edge, the loops are triangulated alone and the points inserted
        allPlanar = np.vstack([planar, candidatePlanar])
        triangles = self._delaunayInLoops(allPlanar, localEdges)
        if triangles is None:
            triangles = self._triangulatePlanarLoops(planar, localEdges)
            if triangles is None:
                return None
            triangles, _hosts = self._insertPoints(allPlanar, triangles, np.arange(loopCount, len(allPlanar)))
        planar = allPlanar
        if len(candidatePlanar) > 0:
            positions = np.vstack([positions, surfacePoints(candidatePlanar)])
            hosts = self._nearestPoints(candidatePlanar, planar[:loopCount])
            for name, values in localArrays.items():
                if name == arrayName:
                    newValues = signedDistances(candidatePlanar)
                else:
                    newValues = values[hosts]  # from the nearest rim point
                localArrays[name] = np.concatenate([values, newValues.astype(values.dtype)])

        # The fallback's ear clipping leaves fans of slivers across the face; later sheets
        # cutting through them would give a needlessly dense, near-degenerate rim
        triangles = self._delaunayFlips(planar, triangles, edgeDeviation)

        # Split cap edges that still stray from the cut surface; the loop edges are never split
        for _iteration in range(self.MAX_REFINE_ITERATIONS):
            count = len(positions)
            triangleEdges = np.stack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=1)
            sortedEdges = np.sort(triangleEdges.reshape(-1, 2), axis=1)
            keys, edgeOfTriangle, edgeUses = np.unique(sortedEdges[:, 0] * count + sortedEdges[:, 1],
                                                       return_inverse=True, return_counts=True)
            start, end = keys // count, keys % count
            midPlanar = (planar[start] + planar[end]) / 2.0
            midPoints = surfacePoints(midPlanar)
            deviation = np.linalg.norm(midPoints - (positions[start] + positions[end]) / 2.0, axis=1)
            lengths = np.linalg.norm(positions[start] - positions[end], axis=1)
            split = (edgeUses == 2) & (deviation > self.CAP_TOLERANCE) & (lengths > self.CAP_MIN_EDGE_LENGTH)
            splitCount = int(np.count_nonzero(split))
            if splitCount == 0:
                break
            # Keep the coarser (still closed) cap rather than refine without end where the chart
            # does not follow the cut surface (e.g. a groove floor whose depth jumps)
            budget = max(self.CAP_MIN_POINT_BUDGET, self.CAP_POINT_BUDGET_PER_RIM_POINT * len(loopIds))
            if count - len(loopIds) + splitCount > budget or firstNewId + count + splitCount > self.MAX_REFINED_POINTS:
                logging.warning(f"OsteotomyCuts: a cut face was not refined further ({count - len(loopIds)} points "
                                "added); it may stray slightly from the cut surface.")
                break
            splitStart, splitEnd = start[split], end[split]
            planar = np.vstack([planar, midPlanar[split]])
            positions = np.vstack([positions, midPoints[split]])
            for name, values in localArrays.items():
                if name == arrayName:
                    newValues = signedDistances(midPlanar[split])
                elif np.issubdtype(values.dtype, np.floating):
                    newValues = (values[splitStart] + values[splitEnd]) / 2.0
                else:
                    newValues = values[splitStart]
                localArrays[name] = np.concatenate([values, newValues.astype(values.dtype)])
            midpointOfEdge = np.full(len(keys), -1, dtype=np.int64)
            midpointOfEdge[split] = count + np.arange(splitCount)
            triangles = self._splitTriangles(triangles, midpointOfEdge[edgeOfTriangle.reshape(-1, 3)])

        # Orient the cap so that it runs along each loop edge opposite to the mesh triangle there
        capEdges = {(int(a), int(b)) for a, b in np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]],
                                                              triangles[:, [2, 0]]])}
        agreeing = sum((int(a), int(b)) in capEdges for a, b in localEdges)
        if 2 * agreeing > len(localEdges):
            triangles = triangles[:, [0, 2, 1]]

        # Drop added points that no triangle uses (bend points outside every triangle)
        loopCount = len(loopIds)
        used = np.zeros(len(positions), dtype=bool)
        used[triangles.ravel()] = True
        used[:loopCount] = True
        if not np.all(used):
            newIndex = np.cumsum(used) - 1
            triangles = newIndex[triangles]
            positions = positions[used]
            localArrays = {name: values[used] for name, values in localArrays.items()}
        meshIds = np.concatenate([loopIds, firstNewId + np.arange(len(positions) - loopCount)])
        addedArrays = {name: values[loopCount:] for name, values in localArrays.items()}
        return positions[loopCount:], meshIds[triangles], addedArrays

    def _delaunayInLoops(self, planar: np.ndarray, edges: np.ndarray) -> Optional[np.ndarray]:
        """Delaunay triangulation of 2D points, restricted to the region bounded by loops.

        The loop points come first in planar. The triangulation (scipy / Qhull, robust to
        collinear points) is not constrained, so it is accepted only if its triangles inside
        the loops are bounded by exactly the loop edges; with points spaced along the loops more
        closely than the other points keep off them, this is the usual case.

        :param planar: (K, 2) loop points followed by the points inside the loops.
        :param edges: (E, 2) directed edges forming closed loops.
        :return: (C, 3) counter-clockwise triangles, or None (scipy missing or a loop edge is
            not an edge of the triangulation).
        """
        try:
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            from scipy.spatial import Delaunay
        except ImportError:
            return None
        # Four frame points keep the loops off the convex hull, where Qhull joins collinear
        # points with flat triangles
        lower, upper = planar.min(axis=0), planar.max(axis=0)
        margin = 0.1 * (upper - lower) + 1.0
        frame = np.array([[lower[0] - margin[0], lower[1] - margin[1]], [upper[0] + margin[0], lower[1] - margin[1]],
                          [upper[0] + margin[0], upper[1] + margin[1]], [lower[0] - margin[0], upper[1] + margin[1]]])
        framed = np.vstack([planar, frame])
        try:
            triangulation = Delaunay(framed)
        except Exception:  # Qhull errors (e.g. all points collinear)
            return None
        triangles = triangulation.simplices.astype(np.int64)
        count = len(framed)
        if len(np.unique(triangles)) != count:
            return None  # a point was dropped (coincident points)
        corners = framed[triangles]
        area = ((corners[:, 1, 0] - corners[:, 0, 0]) * (corners[:, 2, 1] - corners[:, 0, 1])
                - (corners[:, 1, 1] - corners[:, 0, 1]) * (corners[:, 2, 0] - corners[:, 0, 0]))
        triangles[area < 0] = triangles[area < 0][:, [0, 2, 1]]  # counter-clockwise

        # A loop edge with another loop point in its diametral circle (a narrow neck of a jagged
        # rim) is not a Delaunay edge: recover it by flipping the edges that cross it
        loopKeys = np.unique(np.minimum(edges[:, 0], edges[:, 1]) * count + np.maximum(edges[:, 0], edges[:, 1]))
        present = np.unique(np.concatenate([np.minimum(triangles[:, k], triangles[:, (k + 1) % 3]) * count
                                            + np.maximum(triangles[:, k], triangles[:, (k + 1) % 3]) for k in range(3)]))
        missing = loopKeys[~np.isin(loopKeys, present)]
        if len(missing) > 0:
            if len(missing) > self.MAX_RECOVERED_EDGES:
                return None
            triangles = self._recoverEdges(framed, triangles, np.column_stack([missing // count, missing % count]))
            if triangles is None:
                return None

        # Split the triangles into regions separated by loop edges (neighbours share an edge)
        undirected = np.concatenate([np.minimum(triangles[:, k], triangles[:, (k + 1) % 3]) * count
                                     + np.maximum(triangles[:, k], triangles[:, (k + 1) % 3]) for k in range(3)])
        owners = np.tile(np.arange(len(triangles)), 3)
        order = np.argsort(undirected, kind="stable")
        undirected, owners = undirected[order], owners[order]
        shared = np.flatnonzero((undirected[:-1] == undirected[1:]) & ~np.isin(undirected[:-1], loopKeys))
        rows, columns = owners[shared], owners[shared + 1]
        graph = coo_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(triangles), len(triangles)))
        _regionCount, regions = connected_components(graph, directed=False)

        # The face lies on the same side of every loop edge (the loops are directed as the mesh
        # rim): on the left if the loops enclose positive area. The triangle on that side of each
        # loop edge marks its region inside, the one across marks its region outside. No point
        # tests, which fail on the tiny rim edges of a segmented bone.
        start, end = planar[edges[:, 0]], planar[edges[:, 1]]
        faceOnLeft = np.sum(start[:, 0] * end[:, 1] - end[:, 0] * start[:, 1]) >= 0
        directed = np.concatenate([triangles[:, k] * count + triangles[:, (k + 1) % 3] for k in range(3)])
        owner = np.tile(np.arange(len(triangles)), 3)
        order = np.argsort(directed)
        directed, owner = directed[order], owner[order]

        def trianglesWith(keys: np.ndarray) -> Optional[np.ndarray]:
            found = np.searchsorted(directed, keys)
            if np.any(found >= len(directed)) or np.any(directed[np.minimum(found, len(directed) - 1)] != keys):
                return None  # a loop edge is not an edge of the triangulation
            return owner[found]

        left = trianglesWith(edges[:, 0] * count + edges[:, 1])
        right = trianglesWith(edges[:, 1] * count + edges[:, 0])
        if left is None or right is None:
            return None
        inside, outside = (left, right) if faceOnLeft else (right, left)
        insideRegions, outsideRegions = np.unique(regions[inside]), np.unique(regions[outside])
        if np.intersect1d(insideRegions, outsideRegions).size:
            return None  # the loops do not bound a face consistently
        triangles = triangles[np.isin(regions, insideRegions)]
        if len(triangles) == 0 or triangles.max() >= len(planar):
            return None

        # The kept triangles must be bounded by exactly the loop edges
        sortedEdges = np.sort(np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
        keys, uses = np.unique(sortedEdges[:, 0] * count + sortedEdges[:, 1], return_counts=True)
        if not np.array_equal(keys[uses == 1], loopKeys):
            return None
        return triangles

    @staticmethod
    def _recoverEdges(points: np.ndarray, triangles: np.ndarray, missing: np.ndarray) -> Optional[np.ndarray]:
        """Make each given edge an edge of a 2D triangulation by flipping the edges that cross it
        (Sloan's method for constrained triangulations). No point is added; the triangles stay
        counter-clockwise.

        :param points: (K, 2) point coordinates.
        :param triangles: (C, 3) counter-clockwise triangles covering the points' hull.
        :param missing: (M, 2) point index pairs to become edges.
        :return: the triangles, or None if an edge cannot be recovered (e.g. a point lies on it).
        """
        triangles = [list(triangle) for triangle in np.asarray(triangles, dtype=np.int64)]
        edgeTriangles = {}  # (smaller, larger) point index -> triangles using the edge

        def edgeKey(u: int, v: int) -> tuple[int, int]:
            return (u, v) if u < v else (v, u)

        def link(index: int, add: bool) -> None:
            triangle = triangles[index]
            for k in range(3):
                owners = edgeTriangles.setdefault(edgeKey(triangle[k], triangle[(k + 1) % 3]), set())
                owners.add(index) if add else owners.discard(index)

        def orientation(a: int, b: int, c: int) -> float:
            (ax, ay), (bx, by), (cx, cy) = points[a], points[b], points[c]
            return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

        def crosses(u: int, v: int, a: int, b: int) -> bool:
            """Whether segments uv and ab cross at a point inside both."""
            return (orientation(a, b, u) * orientation(a, b, v) < 0
                    and orientation(u, v, a) * orientation(u, v, b) < 0)

        for index in range(len(triangles)):
            link(index, True)
        for a, b in (tuple(int(value) for value in pair) for pair in missing):
            if edgeTriangles.get(edgeKey(a, b)):
                continue
            queue = [key for key, owners in edgeTriangles.items()
                     if owners and a not in key and b not in key and crosses(key[0], key[1], a, b)]
            budget = 100 * len(queue) + 1000
            while queue:
                budget -= 1
                if budget < 0:
                    return None
                u, v = queue.pop(0)
                owners = edgeTriangles.get(edgeKey(u, v))
                if not owners or len(owners) != 2:
                    continue
                first, second = owners
                p = next(vertex for vertex in triangles[first] if vertex not in (u, v))
                q = next(vertex for vertex in triangles[second] if vertex not in (u, v))
                if not crosses(u, v, p, q):  # the quadrilateral u p v q is not convex: try later
                    queue.append((u, v))
                    continue
                link(first, False)
                link(second, False)
                for index, triangle in ((first, [p, u, q]), (second, [q, v, p])):
                    if orientation(*triangle) < 0:
                        triangle = [triangle[0], triangle[2], triangle[1]]
                    triangles[index] = triangle
                    link(index, True)
                if a not in (p, q) and b not in (p, q) and crosses(p, q, a, b):
                    queue.append((p, q))
            if not edgeTriangles.get(edgeKey(a, b)):
                return None
        return np.array(triangles, dtype=np.int64)

    @staticmethod
    def _nearestPoints(queries: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Index of the nearest of the given points for each query point (2D or 3D)."""
        try:
            from scipy.spatial import cKDTree
            return cKDTree(points).query(queries)[1].astype(np.int64)
        except ImportError:
            nearest = np.empty(len(queries), dtype=np.int64)
            for chunk in range(0, len(queries), 256):
                differences = queries[chunk:chunk + 256, np.newaxis, :] - points[np.newaxis, :, :]
                nearest[chunk:chunk + 256] = np.argmin(np.sum(differences * differences, axis=2), axis=1)
            return nearest

    @staticmethod
    def _triangulatePlanarLoops(planar: np.ndarray, edges: np.ndarray) -> Optional[np.ndarray]:
        """Triangulate the region bounded by closed loops of 2D edges (nested loops are holes).

        :param planar: (K, 2) point coordinates.
        :param edges: (E, 2) directed edges forming closed loops.
        :return: (C, 3) triangles in point indices, or None on failure.
        """
        vtkPoints = vtk.vtkPoints()
        vtkPoints.SetData(numpy_support.numpy_to_vtk(np.column_stack([planar, np.zeros(len(planar))]), deep=True))
        lines = vtk.vtkCellArray()
        offsets = np.arange(0, 2 * len(edges) + 1, 2, dtype=np.int64)
        lines.SetData(numpy_support.numpy_to_vtk(offsets, deep=True, array_type=vtk.VTK_ID_TYPE),
                      numpy_support.numpy_to_vtk(np.ascontiguousarray(edges.ravel(), dtype=np.int64), deep=True,
                                                 array_type=vtk.VTK_ID_TYPE))
        loops = vtk.vtkPolyData()
        loops.SetPoints(vtkPoints)
        loops.SetLines(lines)
        # The triangulator needs the normal the outer loops wind around (it fails otherwise);
        # holes wind the other way but enclose less area, so the total signed area tells
        start, end = planar[edges[:, 0]], planar[edges[:, 1]]
        signedArea = 0.5 * np.sum(start[:, 0] * end[:, 1] - end[:, 0] * start[:, 1])
        normal = [0.0, 0.0, 1.0 if signedArea >= 0 else -1.0]
        polys = vtk.vtkCellArray()
        success = vtk.vtkContourTriangulator.TriangulateContours(loops, 0, len(edges), polys, normal)
        if not success or polys.GetNumberOfCells() == 0:
            return None
        connectivity = numpy_support.vtk_to_numpy(polys.GetConnectivityArray()).astype(np.int64)
        polyOffsets = numpy_support.vtk_to_numpy(polys.GetOffsetsArray())
        if np.any(np.diff(polyOffsets) != 3) or connectivity.max() >= len(planar):
            return None  # not plain triangles on the given points
        return connectivity.reshape(-1, 3)

    @staticmethod
    def _insideLoops(points: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Even-odd test: which 2D points lie inside the region bounded by the given edges."""
        inside = np.zeros(len(points), dtype=bool)
        for chunk in range(0, len(points), 512):
            p = points[chunk:chunk + 512, np.newaxis, :]
            straddles = (starts[np.newaxis, :, 1] > p[..., 1]) != (ends[np.newaxis, :, 1] > p[..., 1])
            with np.errstate(divide="ignore", invalid="ignore"):
                crossingX = starts[:, 0] + (p[..., 1] - starts[:, 1]) * (ends[:, 0] - starts[:, 0]) / (ends[:, 1] - starts[:, 1])
            inside[chunk:chunk + 512] = np.count_nonzero(straddles & (crossingX > p[..., 0]), axis=1) % 2 == 1
        return inside

    @staticmethod
    def _distanceToSegments(points: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Distance of each 2D point to the nearest of the given segments."""
        result = np.full(len(points), np.inf)
        direction = ends - starts
        lengthSquared = np.maximum(np.sum(direction * direction, axis=1), 1e-300)
        for chunk in range(0, len(points), 512):
            p = points[chunk:chunk + 512, np.newaxis, :]
            t = np.clip(np.sum((p - starts) * direction, axis=2) / lengthSquared, 0.0, 1.0)
            nearest = starts + t[..., np.newaxis] * direction
            result[chunk:chunk + 512] = np.linalg.norm(p - nearest, axis=2).min(axis=1)
        return result

    def _gradedFillPoints(self, starts: np.ndarray, ends: np.ndarray, features: np.ndarray,
                          spacing: float) -> np.ndarray:
        """Points filling the 2D region bounded by loops, spaced more widely away from them.

        A point is placed on a square grid of spacing h * 2^k where its distance to the nearest
        loop edge or feature point is between h * 2^k and h * 2^(k + 1) (h = spacing, and
        h * 2^k at most CAP_MAX_FILL_SPACING), so the spacing grows with the distance and no
        point comes closer than h to the loops or the features. Distances come from a Euclidean
        distance transform on a raster of pixel h / 2 (coarser for very large faces).

        :param starts: (E, 2) loop edge starts.
        :param ends: (E, 2) loop edge ends.
        :param features: (F, 2) points already placed inside the loops (along bends).
        :param spacing: point spacing h next to the loops (> 0).
        :return: (K, 2) points inside the loops; none if scipy is not available.
        """
        try:
            from scipy import ndimage  # bundled with Slicer
        except ImportError:
            return np.zeros((0, 2))  # the cap is still closed, only less regular
        lower = np.minimum(starts, ends).min(axis=0)
        extent = np.maximum(starts, ends).max(axis=0) - lower
        pixel = spacing / 2.0
        pixelCount = float(np.prod(np.floor(extent / pixel) + 1))
        if pixelCount > self.MAX_FILL_PIXELS:
            pixel *= np.sqrt(pixelCount / self.MAX_FILL_PIXELS)
        baseStep = max(1, int(round(spacing / pixel)))  # grid step (pixels) next to the loops
        nx, ny = (np.floor(extent / pixel).astype(int) + 1).tolist()
        xs, ys = lower[0] + pixel * np.arange(nx), lower[1] + pixel * np.arange(ny)

        # Distance of each pixel centre to the loops (sampled at half a pixel) and features
        lengths = np.linalg.norm(ends - starts, axis=1)
        counts = np.ceil(lengths / (pixel / 2.0)).astype(int) + 1
        edgeIds = np.repeat(np.arange(len(starts)), counts)
        within = np.arange(len(edgeIds)) - np.repeat(np.cumsum(counts) - counts, counts)
        t = within / np.maximum(np.repeat(counts, counts) - 1, 1)
        samples = np.vstack([starts[edgeIds] + t[:, np.newaxis] * (ends - starts)[edgeIds], features])
        cells = np.rint((samples - lower) / pixel).astype(int)
        free = np.ones((nx, ny), dtype=bool)
        free[np.clip(cells[:, 0], 0, nx - 1), np.clip(cells[:, 1], 0, ny - 1)] = False
        # Less a pixel for the rounding of the samples to pixel centres
        clearance = (ndimage.distance_transform_edt(free) - 1.0) / baseStep

        # Even-odd inside test of the pixel centres along rows of constant y (as _insideLoops)
        rowStarts = np.searchsorted(ys, np.minimum(starts[:, 1], ends[:, 1]), "left")
        rowCounts = np.searchsorted(ys, np.maximum(starts[:, 1], ends[:, 1]), "left") - rowStarts
        edgeIds = np.repeat(np.arange(len(starts)), rowCounts)
        rows = rowStarts[edgeIds] + np.arange(len(edgeIds)) - np.repeat(np.cumsum(rowCounts) - rowCounts, rowCounts)
        s, e = starts[edgeIds], ends[edgeIds]
        crossings = s[:, 0] + (ys[rows] - s[:, 1]) * (e[:, 0] - s[:, 0]) / (e[:, 1] - s[:, 1])
        toggles = np.zeros((nx + 1, ny), dtype=np.int32)
        np.add.at(toggles, (np.searchsorted(xs, crossings, "left"), rows), 1)
        inside = np.cumsum(toggles[:nx], axis=0) % 2 == 1

        maxLevel = max(0, int(np.floor(np.log2(max(self.CAP_MAX_FILL_SPACING / spacing, 1.0)))))
        levels = np.clip(np.floor(np.log2(np.maximum(clearance, 1.0))), 0, maxLevel).astype(int)
        steps = baseStep * 2 ** levels
        i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        keep = inside & (clearance >= 1.0) & (i % steps == 0) & (j % steps == 0)
        return lower + pixel * np.column_stack([i[keep], j[keep]]).astype(float)

    @staticmethod
    def _insertPoints(planar: np.ndarray, triangles: np.ndarray, pointIds: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
        """Insert points into a 2D triangulation by splitting their enclosing triangles in three.

        At most one point goes into each triangle per round; the others are placed in a later
        round in one of the new triangles. Orientation is kept. Points outside every triangle
        are not inserted.

        :param planar: (K, 2) coordinates of all points.
        :param triangles: (C, 3) triangles.
        :param pointIds: ids of the points to insert.
        :return: (triangles, for each point the first corner of the triangle it was inserted
            into, or -1 if it was not inserted).
        """
        triangles = np.array(triangles, dtype=np.int64)
        hosts = np.full(len(pointIds), -1, dtype=np.int64)
        pending = np.arange(len(pointIds))
        while len(pending) > 0:
            corners = planar[triangles]  # (C, 3, 2)
            e0, e1 = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
            denominator = e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]
            valid = np.abs(denominator) > 1e-300
            containing = np.full(len(pending), -1, dtype=np.int64)
            for chunk in range(0, len(pending), 256):
                ids = pending[chunk:chunk + 256]
                e2 = planar[pointIds[ids]][:, np.newaxis, :] - corners[np.newaxis, :, 0]
                with np.errstate(divide="ignore", invalid="ignore"):
                    w1 = (e2[..., 0] * e1[:, 1] - e2[..., 1] * e1[:, 0]) / denominator
                    w2 = (e0[:, 0] * e2[..., 1] - e0[:, 1] * e2[..., 0]) / denominator
                    inside = valid & (w1 >= -1e-12) & (w2 >= -1e-12) & (w1 + w2 <= 1.0 + 1e-12)
                found = inside.any(axis=1)
                containing[chunk:chunk + 256] = np.where(found, inside.argmax(axis=1), -1)
            located = containing >= 0
            if not np.any(located):
                break
            # One point per triangle in this round
            order = np.flatnonzero(located)
            _unique, firstOfTriangle = np.unique(containing[order], return_index=True)
            chosen = order[firstOfTriangle]
            hostTriangles = containing[chosen]
            p = pointIds[pending[chosen]]
            a, b, c = triangles[hostTriangles, 0], triangles[hostTriangles, 1], triangles[hostTriangles, 2]
            hosts[pending[chosen]] = a
            triangles[hostTriangles] = np.column_stack([a, b, p])
            triangles = np.vstack([triangles, np.column_stack([b, c, p]), np.column_stack([c, a, p])])
            remaining = np.ones(len(pending), dtype=bool)
            remaining[chosen] = False
            remaining &= located
            pending = pending[remaining]
        return triangles, hosts

    def _delaunayFlips(self, planar: np.ndarray, triangles: np.ndarray,
                       edgeDeviation: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None) -> np.ndarray:
        """Flip interior edges until the 2D triangulation is (constrained) Delaunay.

        Lawson's algorithm, vectorised: in each round, every interior edge whose opposite
        vertex lies inside the circumcircle of a triangle, or that borders a degenerate
        (zero-area) triangle, is replaced by the other diagonal of its two triangles, if that
        gives two proper triangles. Edges sharing no triangle are flipped together. Boundary
        edges (the loops) are never flipped, orientation is kept, no point is added or moved.

        :param planar: (K, 2) point coordinates.
        :param triangles: (C, 3) consistently oriented triangles.
        :param edgeDeviation: optional distance of edge midpoints (point index arrays) from the
            cut surface; a flip must not take the diagonal farther from it than CAP_TOLERANCE
            or the old diagonal (the plane coordinates ignore folds of the sheet).
        :return: (C, 3) triangles.
        """
        triangles = np.array(triangles, dtype=np.int64).reshape(-1, 3)
        if len(triangles) < 2:
            return triangles
        extent = float(np.linalg.norm(planar.max(axis=0) - planar.min(axis=0)))
        areaTolerance = 1e-12 * extent ** 2

        def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
            ab, ac = planar[b] - planar[a], planar[c] - planar[a]
            return ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]

        corners = triangles
        winding = 1.0 if np.sum(orientation(corners[:, 0], corners[:, 1], corners[:, 2])) >= 0 else -1.0
        order = np.arange(3)
        for _round in range(self.MAX_FLIP_ROUNDS):
            # Each interior edge with its two (triangle, local edge) occurrences
            triangleIds = np.repeat(np.arange(len(triangles)), 3)
            local = np.tile(order, len(triangles))
            starts = triangles[triangleIds, local]
            ends = triangles[triangleIds, (local + 1) % 3]
            keys = np.minimum(starts, ends) * len(planar) + np.maximum(starts, ends)
            sortOrder = np.argsort(keys, kind="stable")
            sortedKeys = keys[sortOrder]
            pairStarts = np.flatnonzero((sortedKeys[:-1] == sortedKeys[1:])
                                        & np.append(True, sortedKeys[1:-1] != sortedKeys[:-2]))
            first, second = sortOrder[pairStarts], sortOrder[pairStarts + 1]
            t1, k1, t2, k2 = triangleIds[first], local[first], triangleIds[second], local[second]
            a, b = triangles[t1, k1], triangles[t1, (k1 + 1) % 3]
            c, d = triangles[t1, (k1 + 2) % 3], triangles[t2, (k2 + 2) % 3]
            consistent = (triangles[t2, k2] == b) & (triangles[t2, (k2 + 1) % 3] == a)

            turn = winding * orientation(a, b, c)
            other = winding * orientation(b, a, d)
            degenerate = (turn <= areaTolerance) | (other <= areaTolerance)
            rows = [planar[p] - planar[d] for p in (a, b, c)]
            lifted = [np.sum(row * row, axis=1) for row in rows]
            inCircle = (rows[0][:, 0] * (rows[1][:, 1] * lifted[2] - lifted[1] * rows[2][:, 1])
                        - rows[0][:, 1] * (rows[1][:, 0] * lifted[2] - lifted[1] * rows[2][:, 0])
                        + lifted[0] * (rows[1][:, 0] * rows[2][:, 1] - rows[1][:, 1] * rows[2][:, 0]))
            illegal = winding * inCircle > 1e-12 * (lifted[0] + lifted[1] + lifted[2]) ** 2
            valid = ((winding * orientation(a, d, c) > areaTolerance)
                     & (winding * orientation(d, b, c) > areaTolerance))
            candidates = np.flatnonzero(consistent & (degenerate | illegal) & valid)
            if edgeDeviation is not None and len(candidates) > 0:
                oldDeviation = edgeDeviation(a[candidates], b[candidates])
                newDeviation = edgeDeviation(c[candidates], d[candidates])
                candidates = candidates[newDeviation <= np.maximum(oldDeviation, self.CAP_TOLERANCE)]
            if len(candidates) == 0:
                break

            # Flip only edges that share no triangle with an earlier candidate
            owner = np.full(len(triangles), len(candidates), dtype=np.int64)
            ranks = np.arange(len(candidates))
            np.minimum.at(owner, t1[candidates], ranks)
            np.minimum.at(owner, t2[candidates], ranks)
            chosen = candidates[(owner[t1[candidates]] == ranks) & (owner[t2[candidates]] == ranks)]
            triangles[t1[chosen]] = np.column_stack([a[chosen], d[chosen], c[chosen]])
            triangles[t2[chosen]] = np.column_stack([d[chosen], b[chosen], c[chosen]])
        return triangles

    @staticmethod
    def isLimitedCut(options: CutOptions) -> bool:
        """Return True if the cut may end inside the model: a limited depth, or a set distance
        past the line ends. Such a cut may leave the model in one piece (a groove or a slot)."""
        return options.depth > 0 or options.endExtension > 0

    @classmethod
    def effectiveKerfWidth(cls, options: CutOptions, stopped: bool = False) -> float:
        """Width of bone removed: options.kerfWidth, or LIMITED_IDEAL_KERF_WIDTH for a limited
        cut with the ideal blade (kerf 0), so that the cut ends where the sheet ends. A line
        that stops at earlier lines of its osteotomy (stopped) is limited too."""
        if options.kerfWidth > 0 or not (stopped or cls.isLimitedCut(options)):
            return float(options.kerfWidth)
        return cls.LIMITED_IDEAL_KERF_WIDTH

    @classmethod
    def effectiveRefineEdgeLength(cls, options: CutOptions, stopped: bool = False) -> float:
        """Maximum edge length near the sheet: options.refineEdgeLength, or half the effective
        kerf width when that is 0 (automatic). 0 means no refinement (zero kerf, no explicit
        length)."""
        if options.refineEdgeLength > 0:
            return float(options.refineEdgeLength)
        return cls.effectiveKerfWidth(options, stopped) / 2.0

    @staticmethod
    def stopSides(pathPoints: np.ndarray, earlierSheets: list[vtk.vtkPolyData]) -> list[float]:
        """Side (+1 / -1) of each earlier sheet on which a later line of the osteotomy lies.

        The path is sampled about every millimetre and the median signed distance decides, so
        that points placed on an earlier line (distance 0) do not matter.

        :param pathPoints: (N, 3) points of the later line.
        :param earlierSheets: sheets of the earlier lines.
        :return: one side per earlier sheet.
        :raises ValueError: if the line lies on an earlier sheet.
        """
        points = np.asarray(pathPoints, dtype=float)
        samples = [points[:1]]
        for start, end in zip(points[:-1], points[1:]):
            count = max(1, int(np.ceil(np.linalg.norm(end - start))))
            samples.append(start + np.outer(np.arange(1, count + 1) / count, end - start))
        samples = np.vstack(samples)
        sides = []
        for index, sheet in enumerate(earlierSheets):
            implicitDistance = vtk.vtkImplicitPolyDataDistance()
            implicitDistance.SetInput(sheet)
            values = vtk.vtkDoubleArray()
            implicitDistance.FunctionValue(numpy_support.numpy_to_vtk(samples, deep=True), values)
            median = float(np.median(numpy_support.vtk_to_numpy(values)))
            if abs(median) < 1e-6:
                raise ValueError(_("A later osteotomy line lies on line {index}: move it to one side of that "
                                   "line.").format(index=index + 1))
            sides.append(1.0 if median > 0 else -1.0)
        return sides

    @staticmethod
    def protectedByStops(points: np.ndarray, stops: list[tuple[vtk.vtkPolyData, float]]) -> np.ndarray:
        """Boolean mask of the points on the far side of any stop (earlier sheet, side): a later
        line of an osteotomy does not cut there."""
        protected = np.zeros(len(points), dtype=bool)
        for stopSheet, side in stops:
            implicitDistance = vtk.vtkImplicitPolyDataDistance()
            implicitDistance.SetInput(stopSheet)
            values = vtk.vtkDoubleArray()
            implicitDistance.FunctionValue(numpy_support.numpy_to_vtk(np.ascontiguousarray(points, dtype=float),
                                                                      deep=True), values)
            protected |= side * numpy_support.vtk_to_numpy(values) < 0
        return protected

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
        arrays, normalsName, scalarsName = self._pointArrays(mesh)
        if arrayName not in arrays:
            raise ValueError(_("The mesh has no sheet distance array."))

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
    def _pointArrays(mesh: vtk.vtkPolyData) -> tuple[dict, Optional[str], Optional[str]]:
        """Copies of a mesh's named point data arrays.

        :return: ({name: (numpy values, VTK data type)}, name of the normals array, name of
            the active scalars), for _buildTriangleMesh.
        """
        pointData = mesh.GetPointData()
        arrays = {}
        for i in range(pointData.GetNumberOfArrays()):
            array = pointData.GetArray(i)
            if array is not None and array.GetName():
                arrays[array.GetName()] = (numpy_support.vtk_to_numpy(array).copy(), array.GetDataType())
        normalsName = pointData.GetNormals().GetName() if pointData.GetNormals() else None
        scalarsName = pointData.GetScalars().GetName() if pointData.GetScalars() else None
        return arrays, normalsName, scalarsName

    def mergeCoincidentPoints(self, polyData: vtk.vtkPolyData) -> vtk.vtkPolyData:
        """Join points with identical coordinates, so that split normals do not split the mesh.

        Fragments carry duplicate points along their sharp edges (for flat shading of the cut
        faces); a fragment that is cut again is joined up first. The first point's data is
        kept; triangles that collapse are dropped.

        :param polyData: triangle mesh.
        :return: the mesh itself if no point is duplicated, otherwise a merged copy.
        """
        points = numpy_support.vtk_to_numpy(polyData.GetPoints().GetData())
        _unique, first, inverse = np.unique(points, axis=0, return_index=True, return_inverse=True)
        if len(first) == len(points):
            return polyData
        inverse = inverse.ravel()
        triangles = inverse[numpy_support.vtk_to_numpy(polyData.GetPolys().GetConnectivityArray()).reshape(-1, 3)]
        valid = ((triangles[:, 0] != triangles[:, 1]) & (triangles[:, 1] != triangles[:, 2])
                 & (triangles[:, 2] != triangles[:, 0]))
        arrays, normalsName, scalarsName = self._pointArrays(polyData)
        arrays = {name: (values[first], dataType) for name, (values, dataType) in arrays.items()}
        return self._buildTriangleMesh(points[first].astype(float), triangles[valid].astype(np.int64), arrays,
                                       normalsName, scalarsName)

    def computeDisplayNormals(self, polyData: vtk.vtkPolyData) -> vtk.vtkPolyData:
        """Point normals for display, split at sharp edges so that cut faces are shaded flat.

        Points along edges sharper than NORMALS_FEATURE_ANGLE are duplicated (see
        mergeCoincidentPoints); the geometry and orientation are unchanged.

        :param polyData: triangle mesh.
        :return: a copy with a "Normals" point array.
        """
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(polyData)
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.SplittingOn()
        normals.SetFeatureAngle(self.NORMALS_FEATURE_ANGLE)
        normals.ConsistencyOff()
        normals.AutoOrientNormalsOff()
        normals.Update()
        result = vtk.vtkPolyData()
        result.DeepCopy(normals.GetOutput())
        return result

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
            Sheets cut with a kerf leave a signed distance array "SheetSide<i>" (i = sheet
            index) instead of a split into sides.
        :param sideSignature: side of each zero-kerf sheet this mesh lies on, copied to every
            piece.
        :return: one FragmentPiece per connected region. For each kerf sheet, the side the
            piece mostly lies on (+1 / -1) is appended to its signature, in sheet order.
        """
        if polyData.GetNumberOfCells() == 0:
            return []
        regions, _pointRegions, cellRegions, _regionCount = self._connectedRegions(polyData)
        pieces = []
        for piece in self._splitByCellLabel(regions, cellRegions).values():
            pointData = piece.GetPointData()
            componentIds = numpy_support.vtk_to_numpy(pointData.GetArray("ComponentId"))
            hostIds = numpy_support.vtk_to_numpy(pointData.GetArray("HostComponentId"))
            # A connected piece comes from one component; the mode guards against interpolation noise
            componentId = int(np.bincount(componentIds).argmax())
            hostComponentId = int(hostIds[np.argmax(componentIds == componentId)])
            signature = tuple(sideSignature)
            for sheetIndex in sorted(self._sheetSideIndices(piece)):
                sides = numpy_support.vtk_to_numpy(pointData.GetArray(f"{SHEET_SIDE_PREFIX}{sheetIndex}"))
                signature += (1 if np.count_nonzero(sides > 0) >= np.count_nonzero(sides < 0) else -1,)
            pieces.append(FragmentPiece(piece, componentId, hostComponentId, signature,
                                        piece.GetNumberOfPoints()))
        return pieces

    @staticmethod
    def _sheetSideIndices(polyData: vtk.vtkPolyData) -> list[int]:
        """Sheet indices of the "SheetSide<i>" point arrays of a mesh."""
        pointData = polyData.GetPointData()
        names = (pointData.GetArrayName(i) or "" for i in range(pointData.GetNumberOfArrays()))
        return [int(name[len(SHEET_SIDE_PREFIX):]) for name in names
                if name.startswith(SHEET_SIDE_PREFIX) and name[len(SHEET_SIDE_PREFIX):].isdigit()]

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
                    options: CutOptions | list[CutOptions],
                    progressCallback: Optional[ProgressCallback] = None,
                    stops: Optional[list[list[tuple[vtk.vtkPolyData, float]]]] = None) -> list[vtk.vtkPolyData]:
        """Cut a mesh with one or more sheets into separate fragments (headless core).

        Each sheet is applied in turn to every current piece, then the pieces are split into
        connected fragments and enclosed pieces are merged into their host. Several sheets let
        Phase 3 cut several osteotomies at once. The input mesh is not modified.

        With kerfWidth = 0 each piece is split at signed distance 0 into its two sides. With a
        kerf, material within kerfWidth / 2 of the sheet is removed (removeKerf) and the sides
        are found afterwards from the signed distance kept per sheet. A limited cut (depth or
        end reach set) with kerfWidth = 0 removes LIMITED_IDEAL_KERF_WIDTH (effectiveKerfWidth),
        so that it ends where its sheet ends.

        With capCutFaces, the cut faces a sheet leaves are capped (capCutFaces) right
        after that sheet, so later sheets cut closed meshes and the fragments are watertight.
        A groove left by a depth-limited cut is lined the same way, walls and rounded floor.
        Fragments then get display normals split at sharp edges (computeDisplayNormals).

        Several lines of one osteotomy are cut together with ``stops``: a later line only removes
        bone on the side of each earlier line where its own points lie (stopSides), so that e.g.
        the pterygomaxillary cut of a Le Fort I ends at the horizontal cut. The earlier line's
        kerf gap holds no mesh, so this never splits a triangle and the caps are unchanged. A
        line with stops always uses a kerf (effectiveKerfWidth).

        :param polyData: model mesh in world coordinates.
        :param sheets: cutting sheets from buildSheetPolyData, applied in this order.
        :param options: cut options, one for all sheets or one per sheet. The first gives the
            minimum fragment size.
        :param progressCallback: called with (percent, message) between stages.
        :param stops: per sheet, the (earlier sheet, side) pairs it stops at (empty for the
            first line); None for no stops.
        :return: fragment meshes, largest first. A depth-limited kerf cut may leave one
            fragment (a groove).
        :raises ValueError: if there is no sheet, the mesh has no polygons, a kerf sheet does
            not reach the model, or a kerf sheet ends inside the model beyond its path ends
            (sheetEndsInModel: the rounded ends of such a slot cannot be capped yet), or a
            line meets bone across an earlier line outside that line's cut.
        """
        report = progressCallback or (lambda percent, message: None)
        if not sheets:
            raise ValueError(_("At least one cutting sheet is needed."))
        optionsList = list(options) if isinstance(options, (list, tuple)) else [options] * len(sheets)
        stopsList = list(stops) if stops is not None else [[] for _sheet in sheets]
        if len(optionsList) != len(sheets) or len(stopsList) != len(sheets):
            raise ValueError(_("One set of options and stops is needed per cutting sheet."))
        # A fragment of an earlier cut has duplicate points along sharp edges (display normals)
        triangles = self.mergeCoincidentPoints(self._ensureTriangles(polyData))
        if triangles.GetNumberOfCells() == 0:
            raise ValueError(_("The model has no surface polygons to cut."))
        kerfWidths = [self.effectiveKerfWidth(o, bool(s)) for o, s in zip(optionsList, stopsList)]
        uncapped = 0

        endsMessage = _("Osteotomy line {index}: the cut ends inside the bone beyond the ends of the line. Increase "
                        "\"Past line ends\" until the red outline of the preview no longer ends on the bone, or set "
                        "it to 0.")
        if any(width > 0 and not sheetStops for width, sheetStops in zip(kerfWidths, stopsList)):
            locator = vtk.vtkCellLocator()
            locator.SetDataSet(triangles)
            locator.BuildLocator()
            for sheetIndex, sheet in enumerate(sheets):
                if kerfWidths[sheetIndex] > 0 and not stopsList[sheetIndex] and self.sheetEndsInModel(
                        triangles, sheet, kerfWidths[sheetIndex], locator):
                    raise ValueError(endsMessage.format(index=sheetIndex + 1))

        report(5, _("Finding internal shells..."))
        labelled = self.labelEnclosedComponents(triangles, optionsList[0].minFragmentFraction)
        sides = [(labelled, ())]
        for sheetIndex, sheet in enumerate(sheets):
            report(15 + int(60 * sheetIndex / len(sheets)), _("Cutting..."))
            sheetOptions, sheetStops = optionsList[sheetIndex], stopsList[sheetIndex]
            kerfWidth = kerfWidths[sheetIndex]
            halfKerf = kerfWidth / 2.0
            cap = sheetOptions.capCutFaces
            if sheetStops:  # the ends are checked on the pieces left by the earlier lines
                for mesh, _signature in sides:
                    if self.sheetEndsInModel(mesh, sheet, kerfWidth, stops=sheetStops):
                        raise ValueError(endsMessage.format(index=sheetIndex + 1))
            nextSides = []
            removedAny = False
            sheetMap = SheetParameterisation(sheet, halfKerf) if cap else None
            for mesh, signature in sides:
                withDistance = self.computeSheetDistance(mesh, sheet)
                maxEdgeLength = self.effectiveRefineEdgeLength(sheetOptions, bool(sheetStops))
                if maxEdgeLength > 0:
                    withDistance = self.refineNearSheet(withDistance, sheet, maxEdgeLength,
                                                        halfKerf + maxEdgeLength)
                if kerfWidth > 0:
                    sideArray = vtk.vtkDoubleArray()
                    sideArray.DeepCopy(withDistance.GetPointData().GetArray("SheetDistance"))
                    sideArray.SetName(f"{SHEET_SIDE_PREFIX}{sheetIndex}")
                    withDistance.GetPointData().AddArray(sideArray)
                    protectedName = None
                    if sheetStops:
                        protectedName = STOP_PROTECTED_ARRAY
                        meshPoints = numpy_support.vtk_to_numpy(withDistance.GetPoints().GetData())
                        self._addIntPointArray(withDistance, protectedName,
                                               self.protectedByStops(meshPoints, sheetStops))
                    try:
                        remaining, removed = self.removeKerf(withDistance, sheet, kerfWidth,
                                                             protectedName=protectedName)
                    except StopConflictError:
                        raise ValueError(_("Osteotomy line {index} reaches bone on the far side of an earlier line, "
                                           "beyond the end of that line's cut. Make the earlier line deeper (cut "
                                           "depth) or longer (past line ends), or move line {index} so that it meets "
                                           "the earlier cut.").format(index=sheetIndex + 1))
                    if protectedName:
                        remaining.GetPointData().RemoveArray(protectedName)
                    removedAny = removedAny or removed
                    if cap and removed:
                        remaining, failed = self.capCutFaces(remaining, sheetMap)
                        uncapped += failed
                    if remaining.GetNumberOfCells() > 0:
                        nextSides.append((remaining, signature))
                    continue
                positive, negative = self.splitByDistance(withDistance, sheetOptions)
                for part, side in ((positive, 1), (negative, -1)):
                    if part.GetNumberOfCells() > 0:
                        if cap:
                            part, failed = self.capCutFaces(part, sheetMap)
                            uncapped += failed
                        nextSides.append((part, signature + (side,)))
            if kerfWidth > 0 and not removedAny:
                raise ValueError(_("Cutting sheet {index} does not reach the model. Check the cut path, "
                                   "direction and depth.").format(index=sheetIndex + 1))
            sides = nextSides

        report(75, _("Separating fragments..."))
        pieces = []
        for mesh, signature in sides:
            pieces.extend(self.extractFragments(mesh, signature))
        fragments = self.mergeEnclosedPieces(pieces, optionsList[0].minFragmentFraction * labelled.GetNumberOfPoints())

        for fragment in fragments:
            sideArrays = [f"{SHEET_SIDE_PREFIX}{i}" for i in self._sheetSideIndices(fragment)]
            for arrayName in ["ComponentId", "HostComponentId", "SheetDistance"] + sideArrays:
                fragment.GetPointData().RemoveArray(arrayName)
        if uncapped:
            logging.warning(f"OsteotomyCuts: {uncapped} cut face(s) could not be capped; "
                            "the fragments are not watertight there.")
        if any(o.capCutFaces for o in optionsList):
            fragments = [self.computeDisplayNormals(fragment) for fragment in fragments]
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

    def countOpenEdges(self, polyData: vtk.vtkPolyData) -> tuple[int, int]:
        """(boundary edges, non-manifold edges) of a mesh, after merging coincident points
        (fragments have duplicate points along sharp edges for display). (0, 0) means closed."""
        counts = []
        merged = self.mergeCoincidentPoints(self._ensureTriangles(polyData))
        for boundary in (True, False):
            edges = vtk.vtkFeatureEdges()
            edges.SetInputData(merged)
            edges.SetBoundaryEdges(boundary)
            edges.SetNonManifoldEdges(not boundary)
            edges.FeatureEdgesOff()
            edges.ManifoldEdgesOff()
            edges.Update()
            counts.append(edges.GetOutput().GetNumberOfCells())
        return counts[0], counts[1]

    def assessModel(self, polyData: vtk.vtkPolyData) -> ModelQuality:
        """Check whether a bone model is closed, and count its pieces and internal shells.

        A model that is not closed may not separate, or give bone segments that are not
        closed; internal shells and loose pieces are cut too (use "Treat bone as solid").

        :param polyData: bone mesh (world coordinates).
        :return: the counts; ModelQuality.isClosed tells whether it is closed.
        """
        triangles = self.mergeCoincidentPoints(self._ensureTriangles(polyData))
        openEdges, nonManifoldEdges = self.countOpenEdges(triangles)
        labelled = self.labelEnclosedComponents(triangles, CutOptions().minFragmentFraction)
        components = numpy_support.vtk_to_numpy(labelled.GetPointData().GetArray("ComponentId"))
        hosts = numpy_support.vtk_to_numpy(labelled.GetPointData().GetArray("HostComponentId"))
        pieceCount = len(np.unique(components)) if len(components) else 0
        enclosed = len(np.unique(components[hosts >= 0])) if len(components) else 0
        return ModelQuality(openEdges, nonManifoldEdges, pieceCount, enclosed)

    def assessModelToCut(self, parameterNode: OsteotomyCutsParameterNode) -> Optional[ModelQuality]:
        """Quality of the model a cut would use, or None when it is made solid for the cut
        (treatBoneAsSolid with scipy and a model not marked solid: the solid copy is closed).

        :raises ValueError: if the model is empty or under a non-linear transform.
        """
        inputModel = parameterNode.inputModel
        if (parameterNode.treatBoneAsSolid and importNdimage() is not None
                and not self.isSolidModel(inputModel)):
            return None
        return self.assessModel(self.getWorldPolyData(inputModel))

    def assessSegments(self, nodes: list[vtkMRMLModelNode]) -> list[SegmentQuality]:
        """Check each bone segment for being closed and record it on the node: attributes
        OsteotomyCuts.Watertight ("true" / "false") and OsteotomyCuts.Volume_mm3.

        :param nodes: bone segment model nodes.
        :return: one result per node, in order.
        """
        results = []
        for node in nodes:
            polyData = node.GetPolyData()
            openEdges, nonManifoldEdges = self.countOpenEdges(polyData)
            closed = openEdges == 0 and nonManifoldEdges == 0
            massProperties = vtk.vtkMassProperties()
            massProperties.SetInputData(self._ensureTriangles(polyData))
            massProperties.Update()
            volume = float(massProperties.GetVolume())
            node.SetAttribute("OsteotomyCuts.Watertight", "true" if closed else "false")
            node.SetAttribute("OsteotomyCuts.Volume_mm3", f"{volume:.1f}")
            results.append(SegmentQuality(node.GetName(), closed, volume))
        return results

    def joinSmallSegments(self, fragments: list[vtk.vtkPolyData], minFraction: float,
                          contactDistance: float = 1.0) -> tuple[list[vtk.vtkPolyData], int]:
        """Join small pieces left between the lines of an osteotomy to a large bone segment.

        Cuts through thin bone (pterygoid plates, sinus and nasal walls), and several lines
        meeting, leave loose pieces that surgery would not separate. A piece with less surface
        area than minFraction of the largest segment is added (as a separate shell of the same
        model) to the large segment it touches most: the one with most of its points within
        contactDistance, or else the nearest. Surface area is used as it is meaningful for open
        meshes too.

        :param fragments: bone segments, largest first (cutPolyData output).
        :param minFraction: fraction of the largest segment's surface area; 0 joins nothing.
        :param contactDistance: points closer than this (mm) to a large segment touch it.
        :return: (bone segments, largest first, number of pieces joined).
        """
        if minFraction <= 0 or len(fragments) < 3:
            return fragments, 0

        def area(mesh: vtk.vtkPolyData) -> float:
            massProperties = vtk.vtkMassProperties()
            massProperties.SetInputData(self._ensureTriangles(mesh))
            massProperties.Update()
            return massProperties.GetSurfaceArea()

        areas = np.array([area(fragment) for fragment in fragments])
        isLarge = areas >= minFraction * areas.max()
        large = [i for i in range(len(fragments)) if isLarge[i]]
        if len(large) == len(fragments) or len(large) == 0:
            return fragments, 0
        distanceFunctions = {}
        for i in large:
            distanceFunctions[i] = vtk.vtkImplicitPolyDataDistance()
            distanceFunctions[i].SetInput(fragments[i])
        groups = {i: [fragments[i]] for i in large}
        for j in (j for j in range(len(fragments)) if not isLarge[j]):
            points = fragments[j].GetPoints().GetData()
            contacts, nearest = [], []
            for i in large:
                values = vtk.vtkDoubleArray()
                distanceFunctions[i].FunctionValue(points, values)
                distances = np.abs(numpy_support.vtk_to_numpy(values))
                contacts.append(int(np.count_nonzero(distances <= contactDistance)))
                nearest.append(float(distances.min()) if len(distances) else np.inf)
            best = large[int(np.argmax(contacts))] if max(contacts) > 0 else large[int(np.argmin(nearest))]
            groups[best].append(fragments[j])
        joined = []
        for i in large:
            append = vtk.vtkAppendPolyData()  # no point merging: keeps the split display normals
            for mesh in groups[i]:
                append.AddInputData(mesh)
            append.Update()
            mesh = vtk.vtkPolyData()
            mesh.DeepCopy(append.GetOutput())
            joined.append((areas[i], mesh))
        joined.sort(key=lambda item: item[0], reverse=True)
        return [mesh for _area, mesh in joined], len(fragments) - len(large)

    def sheetEndsInModel(self, polyData: vtk.vtkPolyData, sheetPolyData: vtk.vtkPolyData, kerfWidth: float,
                         locator: Optional[vtk.vtkCellLocator] = None,
                         stops: Optional[list[tuple[vtk.vtkPolyData, float]]] = None) -> bool:
        """Return True if an end of an open sheet lies in the model, so that a kerf cut would
        end inside the bone beyond the path ends (a slot).

        The ends are the sheet's first and last rulings (from the outer to the inner edge). An
        end lies in the model if it passes within half the kerf of the model surface: points
        spaced half a kerf apart along it are tested within a slightly larger radius, so an end
        a little farther away may also count (never one that is closer).

        :param polyData: triangulated model mesh.
        :param sheetPolyData: cutting sheet from buildSheetPolyData.
        :param kerfWidth: width of bone removed (mm).
        :param locator: vtkCellLocator built on polyData; built here if None.
        :param stops: (earlier sheet, side) pairs of an osteotomy line that stops at earlier
            lines: end samples, and model points found, beyond a stop are ignored.
        :return: False for a closed sheet or ends clear of the model.
        """
        sheetPoints = numpy_support.vtk_to_numpy(sheetPolyData.GetPoints().GetData()).astype(float)
        pairCount = len(sheetPoints) // 2
        if sheetPolyData.GetNumberOfPolys() == 2 * pairCount:
            return False  # closed: no ends
        if locator is None:
            locator = vtk.vtkCellLocator()
            locator.SetDataSet(polyData)
            locator.BuildLocator()
        halfKerf = kerfWidth / 2.0
        radius = halfKerf * np.sqrt(1.0 + 0.25 ** 2)  # covers the gaps between the samples
        closest = [0.0, 0.0, 0.0]
        cellId, subId, distance2 = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
        for pair in (0, pairCount - 1):
            start, end = sheetPoints[2 * pair], sheetPoints[2 * pair + 1]
            count = int(np.ceil(np.linalg.norm(end - start) / halfKerf)) + 1
            samples = np.linspace(start, end, count)
            if stops:
                samples = samples[~self.protectedByStops(samples, stops)]
            for point in samples:
                if locator.FindClosestPointWithinRadius(point.tolist(), radius, closest, cellId, subId, distance2):
                    if stops and self.protectedByStops(np.array([closest]), stops)[0]:
                        continue  # bone beyond an earlier line: this line does not cut it
                    return True
        return False

    def computeCutOutline(self, polyData: vtk.vtkPolyData, sheetPolyData: vtk.vtkPolyData,
                          locator: Optional[vtk.vtkStaticCellLocator] = None,
                          stops: Optional[list[tuple[vtk.vtkPolyData, float]]] = None) -> vtk.vtkPolyData:
        """Return the lines where the cutting sheet meets the model surface (cut outline).

        Exact triangle-triangle intersection: for each sheet triangle, the model triangles
        crossing its plane are found with a cell locator plane query (fast enough for live
        preview on large bones), each is intersected with the plane, and the segment is clipped
        to the sheet triangle. So the outline ends exactly where the sheet ends.

        :param polyData: triangulated model mesh in world coordinates.
        :param sheetPolyData: cutting sheet from buildSheetPolyData.
        :param locator: vtkStaticCellLocator built on polyData; built here if None.
        :param stops: (earlier sheet, side) pairs of a later osteotomy line: segments beyond
            them are left out, as the line does not cut there.
        :return: line segments on the model surface (no lines if the sheet misses the model).
        """
        empty = vtk.vtkPolyData()
        if polyData.GetNumberOfPolys() == 0 or sheetPolyData.GetNumberOfPolys() == 0:
            return empty
        if locator is None:
            locator = vtk.vtkStaticCellLocator()
            locator.SetDataSet(polyData)
            locator.BuildLocator()
        points = numpy_support.vtk_to_numpy(polyData.GetPoints().GetData()).astype(float)
        triangles = numpy_support.vtk_to_numpy(polyData.GetPolys().GetConnectivityArray()).reshape(-1, 3)
        sheetPoints = numpy_support.vtk_to_numpy(sheetPolyData.GetPoints().GetData()).astype(float)
        sheetTriangles = numpy_support.vtk_to_numpy(sheetPolyData.GetPolys().GetConnectivityArray()).reshape(-1, 3)

        segments = []
        found = vtk.vtkIdList()
        for corners in sheetPoints[sheetTriangles]:
            normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
            length = np.linalg.norm(normal)
            if not length > 0:
                continue
            normal /= length
            found.Reset()
            locator.FindCellsAlongPlane(corners[0].tolist(), normal.tolist(), 0.0, found)
            cellIds = np.array([found.GetId(i) for i in range(found.GetNumberOfIds())], dtype=np.int64)
            if len(cellIds) == 0:
                continue
            cellPoints = points[triangles[cellIds]]  # (k, 3 corners, xyz)
            overlaps = (np.all(cellPoints.max(axis=1) >= corners.min(axis=0), axis=1)
                        & np.all(cellPoints.min(axis=1) <= corners.max(axis=0), axis=1))
            cellPoints = cellPoints[overlaps]

            # Intersect each model triangle with the plane: the corner alone on its side, and
            # the crossings on its two edges (a corner on the plane counts as below it)
            heights = (cellPoints - corners[0]) @ normal
            above = heights > 0
            aboveCount = above.sum(axis=1)
            crossing = (aboveCount == 1) | (aboveCount == 2)
            cellPoints, heights, above = cellPoints[crossing], heights[crossing], above[crossing]
            if len(cellPoints) == 0:
                continue
            lone = np.where(above.sum(axis=1) == 1, np.argmax(above, axis=1), np.argmin(above, axis=1))
            rows = np.arange(len(cellPoints))
            ends = []
            for other in (lone + 1) % 3, (lone + 2) % 3:
                h0, h1 = heights[rows, lone], heights[rows, other]
                t = (h0 / (h0 - h1))[:, np.newaxis]
                ends.append(cellPoints[rows, lone] + t * (cellPoints[rows, other] - cellPoints[rows, lone]))
            first, second = ends

            # Clip the segments to the sheet triangle (Cyrus-Beck); inward edge normals
            low, high = np.zeros(len(first)), np.ones(len(first))
            step = second - first
            for k in range(3):
                inward = np.cross(normal, corners[(k + 1) % 3] - corners[k])
                value, rate = (first - corners[k]) @ inward, step @ inward
                outside = (np.abs(rate) <= 1e-15) & (value < 0)
                low[outside], high[outside] = 1.0, 0.0
                moving = np.abs(rate) > 1e-15
                limit = np.where(moving, -value / np.where(moving, rate, 1.0), 0.0)
                entering, leaving = moving & (rate > 0), moving & (rate < 0)
                low[entering] = np.maximum(low[entering], limit[entering])
                high[leaving] = np.minimum(high[leaving], limit[leaving])
            keep = high > low
            segments.append(np.stack([first[keep] + low[keep, np.newaxis] * step[keep],
                                      first[keep] + high[keep, np.newaxis] * step[keep]], axis=1))
        segments = np.concatenate(segments) if segments else np.zeros((0, 2, 3))
        if stops and len(segments) > 0:
            segments = segments[~self.protectedByStops(segments.mean(axis=1), stops)]
        if len(segments) == 0:
            return empty

        outline = vtk.vtkPolyData()
        outlinePoints = vtk.vtkPoints()
        outlinePoints.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(segments.reshape(-1, 3)), deep=True))
        outline.SetPoints(outlinePoints)
        cells = vtk.vtkCellArray()
        cells.SetData(numpy_support.numpy_to_vtk(np.arange(0, 2 * len(segments) + 1, 2, dtype=np.int64), deep=True,
                                                 array_type=vtk.VTK_ID_TYPE),
                      numpy_support.numpy_to_vtk(np.arange(2 * len(segments), dtype=np.int64), deep=True,
                                                 array_type=vtk.VTK_ID_TYPE))
        outline.SetLines(cells)
        clean = vtk.vtkCleanPolyData()  # join the segments at shared ends
        clean.SetInputData(outline)
        clean.ToleranceIsAbsoluteOn()
        clean.SetAbsoluteTolerance(0.0)
        clean.Update()
        result = vtk.vtkPolyData()
        result.DeepCopy(clean.GetOutput())
        return result

    def makeSolidPolyData(self, polyData: vtk.vtkPolyData, voxelSize: float = 0.25, closingMm: float = 1.5,
                          smoothingIterations: int = 20,
                          progressCallback: Optional[ProgressCallback] = None) -> vtk.vtkPolyData:
        """Return a closed, solid version of a bone surface (the input is not modified).

        Segmented bone models often have holes, internal surfaces (marrow, canals, the inner
        side of the cortex) and loose specks, so a cut does not separate them cleanly. Here the
        surface is voxelised: the inside found by a scan-line stencil, plus every voxel the
        surface passes through (so the shell stays closed where the mesh has holes). Gaps are
        then sealed by a morphological closing with a ball of radius closingMm (exact Euclidean
        distance transforms), enclosed cavities are filled, pieces smaller than
        SOLID_MIN_PIECE_FRACTION of the largest are dropped, and the surface is rebuilt
        (flying edges) and smoothed (windowed sinc). Surface voxels outside the stencil are
        peeled where bone lies behind them, so the surface is not moved outwards on average;
        plates thinner than a voxel are kept, a voxel thick. The surface moves by up to about
        half a voxel.

        :param polyData: bone mesh in world coordinates.
        :param voxelSize: voxel edge (mm); smaller keeps more detail but needs more memory.
        :param closingMm: radius (mm) of the closing; holes up to about twice this are sealed.
        :param smoothingIterations: iterations of the windowed sinc smoothing (0 = none).
        :param progressCallback: called with (percent, message) between stages.
        :return: closed triangle mesh with outward point normals.
        :raises ValueError: if scipy is missing, the mesh has no polygons, or the volume would
            need more than MAX_SOLID_VOXELS voxels (a coarser voxel size is suggested).
        """
        report = progressCallback or (lambda percent, message: None)
        ndimage = importNdimage()
        if ndimage is None:
            raise ValueError(_("Solid bone models need scipy, which is missing from this Slicer installation."))
        if not voxelSize > 0 or closingMm < 0:
            raise ValueError(_("The solid model detail must be positive and the gap sealing not negative."))
        triangles = self._ensureTriangles(polyData)
        if triangles.GetNumberOfPolys() == 0:
            raise ValueError(_("The model has no surface polygons."))

        # Working volume: the bounds plus room for the closing and a background border
        margin = closingMm + 2.0 * voxelSize
        bounds = np.array(triangles.GetBounds())
        origin = bounds[0::2] - margin
        dimensions = np.ceil((bounds[1::2] + margin - origin) / voxelSize).astype(np.int64) + 1
        voxelCount = int(np.prod(dimensions))
        if voxelCount > self.MAX_SOLID_VOXELS:
            suggested = voxelSize * (voxelCount / self.MAX_SOLID_VOXELS) ** (1.0 / 3.0)
            raise ValueError(_("The model is too large for a solid model with {size:.2f} mm detail ({count} million "
                               "voxels, at most {limit} million). Use a detail of {suggested:.2f} mm or more.").format(
                size=voxelSize, count=voxelCount // 1_000_000, limit=self.MAX_SOLID_VOXELS // 1_000_000,
                suggested=np.ceil(suggested * 100.0) / 100.0))
        spacing = (voxelSize,) * 3
        shape = tuple(int(n) for n in dimensions[::-1])  # numpy order (z, y, x)

        report(0, _("Making the bone solid: filling..."))
        stencil = vtk.vtkPolyDataToImageStencil()
        stencil.SetInputData(triangles)
        stencil.SetOutputOrigin(*origin)
        stencil.SetOutputSpacing(*spacing)
        stencil.SetOutputWholeExtent(0, int(dimensions[0]) - 1, 0, int(dimensions[1]) - 1, 0, int(dimensions[2]) - 1)
        toImage = vtk.vtkImageStencilToImage()
        toImage.SetInputConnection(stencil.GetOutputPort())
        toImage.SetInsideValue(1)
        toImage.SetOutsideValue(0)
        toImage.SetOutputScalarTypeToUnsignedChar()
        toImage.Update()
        inside = numpy_support.vtk_to_numpy(toImage.GetOutput().GetPointData().GetScalars()).reshape(shape) > 0
        report(15, _("Making the bone solid: surface..."))
        surface = np.zeros(shape, dtype=bool)
        self._markSurfaceVoxels(surface, triangles, origin, voxelSize)
        mask = inside | surface

        report(30, _("Making the bone solid: sealing gaps..."))
        if closingMm > 0:
            mask = self._dilateMask(mask, closingMm, voxelSize, ndimage)
            mask = ~self._dilateMask(~mask, closingMm, voxelSize, ndimage)  # erosion

        report(60, _("Making the bone solid: filling cavities..."))
        background, _count = ndimage.label(~mask)
        borderLabels = np.unique(np.concatenate([background[[0, -1]].ravel(), background[:, [0, -1]].ravel(),
                                                 background[:, :, [0, -1]].ravel()]))
        mask = ~np.isin(background, borderLabels[borderLabels > 0])
        del background
        # Surface voxels outside the stencil would move the surface out by half a voxel on
        # average: peel those on the outside with bone behind them (thin plates stay)
        mask &= ~(surface & ~inside & ndimage.binary_dilation(~mask) & ndimage.binary_dilation(inside & mask))
        del inside, surface
        pieces, pieceCount = ndimage.label(mask)
        if pieceCount == 0:
            raise ValueError(_("The solid model is empty. Use a finer detail."))
        sizes = np.bincount(pieces.ravel())
        sizes[0] = 0
        kept = sizes >= self.SOLID_MIN_PIECE_FRACTION * sizes.max()
        mask = kept[pieces]
        del pieces
        dropped = int(pieceCount - np.count_nonzero(kept))

        report(75, _("Making the bone solid: surface..."))
        image = vtk.vtkImageData()
        image.SetDimensions(*(int(n) for n in dimensions))
        image.SetOrigin(*origin)
        image.SetSpacing(*spacing)
        scalars = numpy_support.numpy_to_vtk(mask.astype(np.uint8).ravel(), deep=True,
                                             array_type=vtk.VTK_UNSIGNED_CHAR)
        image.GetPointData().SetScalars(scalars)
        del mask
        contour = vtk.vtkFlyingEdges3D()
        contour.SetInputData(image)
        contour.SetValue(0, 0.5)
        contour.ComputeNormalsOff()
        contour.ComputeGradientsOff()
        contour.ComputeScalarsOff()
        surfacePort = contour.GetOutputPort()
        if smoothingIterations > 0:
            smoother = vtk.vtkWindowedSincPolyDataFilter()
            smoother.SetInputConnection(surfacePort)
            smoother.SetNumberOfIterations(int(smoothingIterations))
            smoother.BoundarySmoothingOff()
            smoother.FeatureEdgeSmoothingOff()
            smoother.SetFeatureAngle(120.0)  # the right angles of the voxel steps are not features
            smoother.SetPassBand(self.SOLID_SMOOTHING_PASS_BAND)
            smoother.NonManifoldSmoothingOn()
            smoother.NormalizeCoordinatesOn()
            surfacePort = smoother.GetOutputPort()
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(surfacePort)
        normals.SplittingOff()
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.Update()
        result = vtk.vtkPolyData()
        result.DeepCopy(normals.GetOutput())
        report(100, _("Making the bone solid: done."))

        def volume(mesh: vtk.vtkPolyData) -> float:
            massProperties = vtk.vtkMassProperties()
            massProperties.SetInputData(mesh)
            massProperties.Update()
            return massProperties.GetVolume()

        logging.info(f"OsteotomyCuts: solid model with {voxelSize:g} mm voxels, {closingMm:g} mm gap sealing: "
                     f"volume {volume(triangles):.0f} -> {volume(result):.0f} mm3 (the first is only meaningful "
                     f"for a closed input), {result.GetNumberOfPolys()} triangles, {dropped} speck(s) dropped")
        return result

    @staticmethod
    def _markSurfaceVoxels(mask: np.ndarray, triangles: vtk.vtkPolyData, origin: np.ndarray,
                           voxelSize: float) -> None:
        """Set the voxels (mask in (z, y, x) order, voxel centres at origin + index * voxelSize)
        that the triangles pass through, sampling each triangle at most half a voxel apart."""
        limit = np.array(mask.shape[::-1]) - 1
        for samples in OsteotomyCutsLogic._triangleSamples(triangles, 0.5 * voxelSize):
            ijk = np.clip(np.rint((samples - origin) / voxelSize).astype(np.int64), 0, limit)
            mask[ijk[:, 2], ijk[:, 1], ijk[:, 0]] = True

    @staticmethod
    def _triangleSamples(triangles: vtk.vtkPolyData, spacing: float):
        """Yield (K, 3) arrays of points covering a triangle mesh, at most spacing apart (a
        barycentric grid per triangle, corners included), in chunks of bounded size."""
        if triangles.GetNumberOfPolys() == 0:
            return
        points = numpy_support.vtk_to_numpy(triangles.GetPoints().GetData()).astype(float)
        corners = points[numpy_support.vtk_to_numpy(triangles.GetPolys().GetConnectivityArray()).reshape(-1, 3)]
        edgeLengths = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2).max(axis=1)
        steps = np.maximum(np.ceil(edgeLengths / spacing), 1).astype(np.int64)
        for n in np.unique(steps):
            i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
            inside = i + j <= n
            a, b = i[inside] / n, j[inside] / n
            weights = np.stack([1.0 - a - b, a, b], axis=1)  # (m, 3)
            group = corners[steps == n]
            chunk = max(1, 4_000_000 // len(weights))
            for start in range(0, len(group), chunk):
                yield np.einsum("mc,kcx->kmx", weights, group[start:start + chunk]).reshape(-1, 3)

    def _dilateMask(self, mask: np.ndarray, radius: float, voxelSize: float, ndimage) -> np.ndarray:
        """Dilate a (z, y, x) mask with an exact ball of the given radius (mm).

        Euclidean distance transforms on slabs along z, overlapping by the radius, so memory
        stays bounded on large volumes; the result is the same as on the whole volume.
        """
        result = np.zeros_like(mask)
        overlap = int(np.ceil(radius / voxelSize)) + 1
        depth = mask.shape[0]
        slab = max(8, self.SOLID_SLAB_VOXELS // max(1, mask.shape[1] * mask.shape[2]))
        for z0 in range(0, depth, slab):
            z1 = min(z0 + slab, depth)
            low, high = max(0, z0 - overlap), min(depth, z1 + overlap)
            part = mask[low:high]
            if not part.any():
                continue
            if part.all():
                result[z0:z1] = True
                continue
            distances = ndimage.distance_transform_edt(~part, sampling=voxelSize)
            result[z0:z1] = distances[z0 - low:z1 - low] <= radius
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

    #
    # Osteotomy lines: settings per line, lines cut together (MRML)
    #

    # Fields of CutOptions stored with each line
    LINE_OPTION_NAMES = ("extension", "minFragmentFraction", "kerfWidth", "depth", "endExtension", "capCutFaces",
                         "refineEdgeLength")

    def lineSettingsFromParameters(self, parameterNode: OsteotomyCutsParameterNode) -> LineSettings:
        """The current direction and cut options of the parameter node (a copy)."""
        options = CutOptions()
        for name in self.LINE_OPTION_NAMES:
            setattr(options, name, getattr(parameterNode.options, name))
        return LineSettings(parameterNode.directionMode, tuple(float(c) for c in parameterNode.viewDirection),
                            parameterNode.directionLine, options)

    def getLineSettings(self, curveNode: vtkMRMLMarkupsCurveNode) -> Optional[LineSettings]:
        """The settings stored on a cut path (storeLineSettings), or None if it has none."""
        text = curveNode.GetAttribute(LINE_SETTINGS_ATTRIBUTE) if curveNode is not None else None
        if not text:
            return None
        try:
            stored = json.loads(text)
            options = CutOptions()
            for name, value in stored.get("options", {}).items():
                if name in self.LINE_OPTION_NAMES:
                    setattr(options, name, type(getattr(options, name))(value))
            return LineSettings(DirectionMode(stored["directionMode"]),
                                tuple(float(c) for c in stored["viewDirection"]),
                                curveNode.GetNodeReference(DIRECTION_LINE_REFERENCE_ROLE), options)
        except (ValueError, KeyError, TypeError) as error:
            logging.warning(f"OsteotomyCuts: ignoring unreadable settings of {curveNode.GetName()}: {error}")
            return None

    def lineSettingsOf(self, curveNode: vtkMRMLMarkupsCurveNode,
                       parameterNode: OsteotomyCutsParameterNode) -> Optional[LineSettings]:
        """Settings of a line: the parameter node's for the selected cut path, otherwise those
        stored on the line (None if it has none)."""
        if parameterNode.cutCurve is not None and curveNode.GetID() == parameterNode.cutCurve.GetID():
            return self.lineSettingsFromParameters(parameterNode)
        return self.getLineSettings(curveNode)

    def storeLineSettings(self, parameterNode: OsteotomyCutsParameterNode) -> None:
        """Save the current direction and cut options on the selected cut path, so that each
        line of an osteotomy keeps its own (saved with the scene)."""
        curveNode = parameterNode.cutCurve
        if curveNode is None:
            return
        settings = self.lineSettingsFromParameters(parameterNode)
        text = json.dumps({"directionMode": settings.directionMode.value,
                           "viewDirection": list(settings.viewDirection),
                           "options": {name: getattr(settings.options, name) for name in self.LINE_OPTION_NAMES}},
                          sort_keys=True)
        if curveNode.GetAttribute(LINE_SETTINGS_ATTRIBUTE) != text:
            curveNode.SetAttribute(LINE_SETTINGS_ATTRIBUTE, text)
        lineId = settings.directionLine.GetID() if settings.directionLine is not None else None
        if curveNode.GetNodeReferenceID(DIRECTION_LINE_REFERENCE_ROLE) != lineId:
            curveNode.SetNodeReferenceID(DIRECTION_LINE_REFERENCE_ROLE, lineId)

    def loadLineSettings(self, parameterNode: OsteotomyCutsParameterNode) -> bool:
        """Show the settings stored on the selected cut path in the parameter node.

        A path without stored settings keeps the current saw settings. If it has not cut
        anything yet (a new line), its depth, reach past the ends and direction start afresh
        (0, 0, not captured), as they belong to each line; a path that cut with the current
        settings (a scene from an earlier version) keeps them all. They are then stored on it.

        :return: True if the path had settings.
        """
        curveNode = parameterNode.cutCurve
        settings = self.getLineSettings(curveNode) if curveNode is not None else None
        rawNode = parameterNode.parameterNode
        if settings is None:
            if curveNode is not None and not self.getCurveResult(curveNode):
                wasModifying = rawNode.StartModify()
                try:
                    parameterNode.options.depth = 0.0
                    parameterNode.options.endExtension = 0.0
                    parameterNode.viewDirection = NOT_CAPTURED
                    parameterNode.directionLine = None
                finally:
                    rawNode.EndModify(wasModifying)
            self.storeLineSettings(parameterNode)
            return False
        wasModifying = rawNode.StartModify()
        try:
            parameterNode.directionMode = settings.directionMode
            parameterNode.viewDirection = settings.viewDirection
            parameterNode.directionLine = settings.directionLine
            for name in self.LINE_OPTION_NAMES:
                value = getattr(settings.options, name)
                if getattr(parameterNode.options, name) != value:
                    setattr(parameterNode.options, name, value)
        finally:
            rawNode.EndModify(wasModifying)
        return True

    def resolveLineDirection(self, settings: LineSettings) -> np.ndarray:
        """Extrusion direction of a line from its settings (as resolveDirection).

        :raises ValueError: if the view direction is not captured or the line is invalid.
        """
        if settings.directionMode == DirectionMode.LINE:
            if settings.directionLine is None:
                raise ValueError(_("Select a direction line."))
            return self.directionFromLine(settings.directionLine)
        if not isDirectionCaptured(settings.viewDirection):
            raise ValueError(_("Capture a view direction first."))
        return self._normalised(np.array(settings.viewDirection), _("Capture a view direction first."))

    @staticmethod
    def getGroupLines(firstLine: vtkMRMLMarkupsCurveNode) -> list[vtkMRMLMarkupsCurveNode]:
        """Further lines cut together with firstLine, in order (empty for a single line)."""
        if firstLine is None:
            return []
        lines = (firstLine.GetNthNodeReference(GROUP_LINE_REFERENCE_ROLE, i)
                 for i in range(firstLine.GetNumberOfNodeReferences(GROUP_LINE_REFERENCE_ROLE)))
        return [line for line in lines if line is not None]

    def setGroupLines(self, firstLine: vtkMRMLMarkupsCurveNode, lines: list[vtkMRMLMarkupsCurveNode]) -> None:
        """Make lines (in this order) the further lines of firstLine's osteotomy. A line can be
        in one osteotomy only: it leaves any other, and gives up further lines of its own."""
        lines = [line for line in {line.GetID(): line for line in lines}.values()
                 if line.GetID() != firstLine.GetID()]
        lineIds = {line.GetID() for line in lines}
        for other in slicer.util.getNodesByClass("vtkMRMLMarkupsCurveNode"):
            if other.GetID() == firstLine.GetID():
                continue
            if other.GetID() in lineIds:
                other.RemoveNodeReferenceIDs(GROUP_LINE_REFERENCE_ROLE)
            else:
                for line in self.getGroupLines(other):
                    if line.GetID() in lineIds or line.GetID() == firstLine.GetID():
                        self._removeReference(other, GROUP_LINE_REFERENCE_ROLE, line)
        firstLine.RemoveNodeReferenceIDs(GROUP_LINE_REFERENCE_ROLE)
        for line in lines:
            firstLine.AddNodeReferenceID(GROUP_LINE_REFERENCE_ROLE, line.GetID())

    def findFirstLine(self, curveNode: vtkMRMLMarkupsCurveNode) -> vtkMRMLMarkupsCurveNode:
        """The first line of the osteotomy the given line belongs to (itself if it is first)."""
        if curveNode is None:
            return None
        for other in slicer.util.getNodesByClass("vtkMRMLMarkupsCurveNode"):
            if any(line.GetID() == curveNode.GetID() for line in self.getGroupLines(other)):
                return other
        return curveNode

    def getOsteotomyLines(self, curveNode: vtkMRMLMarkupsCurveNode) -> list[vtkMRMLMarkupsCurveNode]:
        """All lines of the osteotomy the given line belongs to, first line first."""
        if curveNode is None:
            return []
        firstLine = self.findFirstLine(curveNode)
        return [firstLine] + self.getGroupLines(firstLine)

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
        lines = self.getOsteotomyLines(curveNode)
        if inputModel.GetNodeReferenceID(CURVE_REFERENCE_ROLE) in {line.GetID() for line in lines}:
            return _("The model to cut was produced by this osteotomy. Select another model or path.")
        try:
            self.resolveDirection(parameterNode)
        except ValueError as error:
            return str(error)
        for line in lines:
            if line.GetID() == curveNode.GetID():
                continue
            minPoints = 3 if self.isClosedCurve(line) else 2
            if line.GetNumberOfControlPoints() < minPoints:
                return _("Place at least {count} points on {line}.").format(count=minPoints, line=line.GetName())
            settings = self.getLineSettings(line)
            if settings is None:
                return _("Select {line} as the cut path once to set its direction and depth.").format(
                    line=line.GetName())
            try:
                self.resolveLineDirection(settings)
            except ValueError as error:
                return "{line}: {error}".format(line=line.GetName(), error=error)
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

    def _getSurfaceLocator(self, modelNode: vtkMRMLModelNode) -> vtk.vtkStaticCellLocator:
        """Cell locator on the model surface in world coordinates, cached until the model changes.
        Its data set (GetDataSet()) is the triangulated world mesh."""
        polyData = modelNode.GetPolyData()
        key = (modelNode.GetID(), polyData.GetMTime() if polyData else 0, self._worldMatrixKey(modelNode))
        if self._surfaceLocatorCache is None or self._surfaceLocatorCache[0] != key:
            locator = vtk.vtkStaticCellLocator()
            locator.SetDataSet(self._ensureTriangles(self.getWorldPolyData(modelNode)))
            locator.BuildLocator()
            self._surfaceLocatorCache = (key, locator)
        return self._surfaceLocatorCache[1]

    @staticmethod
    def _worldMatrixKey(modelNode: vtkMRMLModelNode) -> Optional[tuple]:
        """The model's linear transform to world as a tuple (None without one), for cache keys."""
        transformNode = modelNode.GetParentTransformNode()
        if transformNode is None or not transformNode.IsTransformToWorldLinear():
            return None
        matrix = vtk.vtkMatrix4x4()
        transformNode.GetMatrixTransformToWorld(matrix)
        return tuple(matrix.GetElement(r, c) for r in range(4) for c in range(4))

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
        return self.buildLineSheet(parameterNode.inputModel, parameterNode.cutCurve,
                                   self.lineSettingsFromParameters(parameterNode))

    def buildLineSheet(self, modelNode: vtkMRMLModelNode, curveNode: vtkMRMLMarkupsCurveNode,
                       settings: LineSettings) -> vtk.vtkPolyData:
        """Build the cutting sheet of one line from its settings.

        :raises ValueError: if the line or its direction is invalid.
        """
        options = settings.options
        return self.buildSheetPolyData(self.getPathPoints(curveNode), self.resolveLineDirection(settings),
                                       self.computeModelExtent(modelNode, options),
                                       closed=self.isClosedCurve(curveNode),
                                       depth=options.depth if options.depth > 0 else None,
                                       endExtension=options.endExtension if options.endExtension > 0 else None)

    def buildOsteotomy(self, parameterNode: OsteotomyCutsParameterNode,
                       skipInvalid: bool = False) -> list[OsteotomyLine]:
        """Build the sheets of every line of the selected cut path's osteotomy, in order, with
        the earlier lines each one stops at.

        :param parameterNode: module parameters (the selected line's settings come from here).
        :param skipInvalid: leave out lines that cannot be built (for the preview) instead of
            raising.
        :return: the lines, first line first.
        :raises ValueError: if the inputs are invalid (unless skipInvalid).
        """
        if not skipInvalid:
            reason = self.validateInputs(parameterNode)
            if reason:
                raise ValueError(reason)
        built = []
        for curveNode in self.getOsteotomyLines(parameterNode.cutCurve):
            try:
                settings = self.lineSettingsOf(curveNode, parameterNode)
                if settings is None:
                    raise ValueError(_("Select {line} as the cut path once to set its direction and depth.").format(
                        line=curveNode.GetName()))
                sheet = self.buildLineSheet(parameterNode.inputModel, curveNode, settings)
                earlier = [line.sheet for line in built]
                sides = self.stopSides(self.getPathPoints(curveNode), earlier) if earlier else []
            except ValueError:
                if skipInvalid:
                    continue
                raise
            built.append(OsteotomyLine(curveNode, sheet, settings.options, list(zip(earlier, sides))))
        return built

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
            self.buildSheetForParameters(parameterNode)  # the selected line must be valid
            lines = self.buildOsteotomy(parameterNode, skipInvalid=True)
        except ValueError:
            self.hideSheetModel(parameterNode)
            return None
        sheet = lines[0].sheet if len(lines) == 1 else self.mergePolyData([line.sheet for line in lines])

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
        self.updateOutlineModel(parameterNode, lines)
        return sheetNode

    def updateOutlineModel(self, parameterNode: OsteotomyCutsParameterNode,
                           lines: list[OsteotomyLine] | vtk.vtkPolyData) -> vtkMRMLModelNode:
        """Show where the cutting sheets meet the input model's surface (computeCutOutline).

        Creates the "CutOutline" model node on first use (red lines, hidden from node selectors,
        not selectable) and stores it in the parameter node. It shows every place the cut comes
        out of the bone, including places far from the cut path; for later lines of an
        osteotomy, not beyond the earlier lines they stop at.

        :param parameterNode: module parameters; the input model must be valid.
        :param lines: the osteotomy lines shown in the preview (or a single sheet).
        :return: the outline model node.
        """
        if isinstance(lines, vtk.vtkPolyData):
            lines = [OsteotomyLine(parameterNode.cutCurve, lines, parameterNode.options, [])]
        locator = self._getSurfaceLocator(parameterNode.inputModel)
        outlines = [self.computeCutOutline(locator.GetDataSet(), line.sheet, locator, line.stops) for line in lines]
        outlines = [outline for outline in outlines if outline.GetNumberOfPoints() > 0]
        outline = (outlines[0] if len(outlines) == 1 else
                   self.mergePolyData(outlines) if outlines else vtk.vtkPolyData())
        outlineNode = parameterNode.outlineModel
        if outlineNode is None:
            outlineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "CutOutline")
            outlineNode.SetHideFromEditors(True)
            outlineNode.SetSelectable(False)
            outlineNode.CreateDefaultDisplayNodes()
            displayNode = outlineNode.GetDisplayNode()
            displayNode.SetColor(1.0, 0.0, 0.0)
            displayNode.SetLineWidth(self.OUTLINE_LINE_WIDTH)
            displayNode.SetVisibility2D(False)
            parameterNode.outlineModel = outlineNode
        outlineNode.SetAndObservePolyData(outline)
        outlineNode.GetDisplayNode().SetVisibility(True)
        return outlineNode

    def hideSheetModel(self, parameterNode: OsteotomyCutsParameterNode) -> None:
        """Hide the cutting sheet preview and the cut outline, if there are any."""
        for node in (parameterNode.sheetModel, parameterNode.outlineModel):
            if node is not None and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetVisibility(False)

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
        :return: the new fragment model nodes, largest first (a single node for a groove or a
            slot, a limited cut that does not separate the model).
        :raises ValueError: if inputs are invalid, the sheet does not cut the model, or later
            cuts depend on the previous result of this curve.
        """
        reason = self.validateInputs(parameterNode)
        if reason:
            raise ValueError(reason)
        inputModel = parameterNode.inputModel
        lines = self.buildOsteotomy(parameterNode)
        curveNode = lines[0].curve  # the osteotomy's result belongs to its first line
        self._checkNoDependentCuts(curveNode)

        report = progressCallback or (lambda percent, message: None)
        report(0, _("Preparing..."))
        polyData, isSolid = self.getModelToCut(parameterNode, progressCallback)
        fragments = self.cutPolyData(polyData, [line.sheet for line in lines], [line.options for line in lines],
                                     progressCallback, stops=[line.stops for line in lines])
        # A limited cut, or lines stopping at each other, may only cut a groove or a slot
        # (cutPolyData has checked that each removed bone)
        isLimited = len(lines) > 1 or any(self.isLimitedCut(line.options) for line in lines)
        if len(fragments) < (1 if isLimited else 2):
            raise ValueError(_("The cutting sheet does not divide the model. Check the cut path and direction."))
        widestKerf = max(self.effectiveKerfWidth(line.options, bool(line.stops)) for line in lines)
        fragments, self.lastJoinedPieces = self.joinSmallSegments(
            fragments, parameterNode.minSegmentPercent / 100.0, contactDistance=widestKerf + 1.0)

        report(90, _("Creating fragment models..."))
        self.removeCutResult(curveNode)
        nodes = self.createFragmentNodes(fragments, inputModel, curveNode)
        if isSolid and all(line.options.capCutFaces for line in lines):
            for node in nodes:  # closed pieces of a solid: later cuts use them as they are
                node.SetAttribute(SOLID_ATTRIBUTE, "1")
        report(95, _("Checking the bone segments..."))
        self.lastSegmentQuality = self.assessSegments(nodes)
        report(100, _("Done."))
        return nodes

    @staticmethod
    def isSolidModel(modelNode: vtkMRMLModelNode) -> bool:
        """Return True for a model marked solid (SOLID_ATTRIBUTE), which is cut as it is."""
        return modelNode is not None and modelNode.GetAttribute(SOLID_ATTRIBUTE) == "1"

    def getModelToCut(self, parameterNode: OsteotomyCutsParameterNode,
                      progressCallback: Optional[ProgressCallback] = None) -> tuple[vtk.vtkPolyData, bool]:
        """Return the world mesh that a cut of the input model uses, and whether it is solid.

        With treatBoneAsSolid, a model not marked solid is replaced by its solid version
        (getSolidPolyData, cached). Without scipy the model is used as it is (with a warning).

        :raises ValueError: if the model is empty, under a non-linear transform, or too large
            for a solid model at the chosen detail.
        """
        inputModel = parameterNode.inputModel
        if self.isSolidModel(inputModel):
            return self.getWorldPolyData(inputModel), True
        if not parameterNode.treatBoneAsSolid:
            return self.getWorldPolyData(inputModel), False
        if importNdimage() is None:
            logging.warning("OsteotomyCuts: scipy is missing, so the bone is cut as it is, not as a solid.")
            return self.getWorldPolyData(inputModel), False
        return self.getSolidPolyData(inputModel, parameterNode.solidVoxelSize, parameterNode.solidGapSeal,
                                     progressCallback), True

    def getSolidPolyData(self, modelNode: vtkMRMLModelNode, voxelSize: float, gapSeal: float,
                         progressCallback: Optional[ProgressCallback] = None) -> vtk.vtkPolyData:
        """Return the solid version (makeSolidPolyData) of a model's world mesh.

        Cached per model, keyed by its mesh modification time, transform and the settings, so
        that re-applying a cut does not rebuild it. The cached mesh must not be modified.

        :raises ValueError: as makeSolidPolyData, or getWorldPolyData.
        """
        key = (modelNode.GetPolyData().GetMTime() if modelNode.GetPolyData() else 0,
               self._worldMatrixKey(modelNode), float(voxelSize), float(gapSeal))
        cached = self._solidCache.pop(modelNode.GetID(), None)
        if cached is None or cached[0] != key:
            cached = (key, self.makeSolidPolyData(self.getWorldPolyData(modelNode), voxelSize, gapSeal,
                                                  progressCallback=progressCallback))
        self._solidCache[modelNode.GetID()] = cached  # most recently used last
        while len(self._solidCache) > self.SOLID_CACHE_SIZE:
            del self._solidCache[next(iter(self._solidCache))]
        return cached[1]

    def createSolidModel(self, modelNode: vtkMRMLModelNode, voxelSize: float, gapSeal: float,
                         progressCallback: Optional[ProgressCallback] = None) -> vtkMRMLModelNode:
        """Create "<Name>_solid", a solid version of the model in world coordinates, marked
        solid, next to it in the subject hierarchy with its colour. The original is hidden,
        never modified.

        :raises ValueError: as makeSolidPolyData, or if the model is empty or under a
            non-linear transform.
        """
        if modelNode is None:
            raise ValueError(_("Select a model to cut."))
        solid = vtk.vtkPolyData()
        solid.DeepCopy(self.getSolidPolyData(modelNode, voxelSize, gapSeal, progressCallback))
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{modelNode.GetName()}_solid")
        node.SetAndObservePolyData(solid)
        node.CreateDefaultDisplayNodes()
        node.SetAttribute(SOLID_ATTRIBUTE, "1")
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        shNode.SetItemParent(shNode.GetItemByDataNode(node),
                             shNode.GetItemParent(shNode.GetItemByDataNode(modelNode)))
        displayNode = modelNode.GetDisplayNode()
        if displayNode is not None:
            node.GetDisplayNode().SetColor(displayNode.GetColor())
            displayNode.SetVisibility(False)
        return node

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

    #
    # Structures to protect (nerve canal, tooth roots)
    #

    # Cut surface samples: coarse over the whole cut, then fine near the closest point
    CLEARANCE_COARSE_SPACING = 1.0
    CLEARANCE_FINE_SPACING = 0.1
    CLEARANCE_MAX_FINE_SAMPLES = 400_000
    # Structure centrelines (curves) are resampled this finely (mm)
    CENTRELINE_SPACING = 0.1
    # Status colours of the sheet preview and the closest-point markers
    STATUS_COLOURS = {ClearanceStatus.SAFE: (0.2, 0.75, 0.3), ClearanceStatus.TOO_CLOSE: (1.0, 0.6, 0.0),
                      ClearanceStatus.ENTERS: (0.9, 0.1, 0.1)}

    @staticmethod
    def guessStructureCategory(name: str) -> StructureCategory:
        """Category from a node name: nerve / canal / IAN -> nerve; tooth / teeth / root -> tooth."""
        words = re.findall(r"[a-z]+", name.lower())
        if any(word in ("ian", "nerve", "canal", "alveolar", "mental", "infraorbital") for word in words):
            return StructureCategory.NERVE
        if any(word.startswith(("tooth", "teeth", "root", "dental")) for word in words):
            return StructureCategory.TOOTH
        return StructureCategory.OTHER

    def getProtectedStructure(self, node) -> Optional[ProtectedStructure]:
        """The settings of a structure to protect stored on a node, or None if it is not one."""
        text = node.GetAttribute(STRUCTURE_ATTRIBUTE) if node is not None else None
        if not text:
            return None
        try:
            stored = json.loads(text)
            category = StructureCategory(stored.get("category", StructureCategory.OTHER.value))
            return ProtectedStructure(node, category, float(stored.get("safeDistance", DEFAULT_SAFE_DISTANCES[category])),
                                      float(stored.get("radius", DEFAULT_STRUCTURE_RADIUS)),
                                      bool(stored.get("enabled", True)))
        except (ValueError, TypeError) as error:
            logging.warning(f"OsteotomyCuts: ignoring unreadable structure settings of {node.GetName()}: {error}")
            return None

    def setProtectedStructure(self, structure: ProtectedStructure) -> None:
        """Store a structure's settings on its node (saved with the scene)."""
        structure.node.SetAttribute(STRUCTURE_ATTRIBUTE, json.dumps(
            {"category": structure.category.value, "safeDistance": float(structure.safeDistance),
             "radius": float(structure.radius), "enabled": bool(structure.enabled)}, sort_keys=True))

    def addProtectedStructure(self, node) -> ProtectedStructure:
        """Mark a model or curve as a structure to protect (keeping existing settings), with the
        category guessed from its name and its default safe distance.

        :raises ValueError: for other node types.
        """
        if node is None or not (node.IsA("vtkMRMLModelNode") or node.IsA("vtkMRMLMarkupsCurveNode")):
            raise ValueError(_("A structure to protect must be a model or a curve."))
        structure = self.getProtectedStructure(node)
        if structure is None:
            category = self.guessStructureCategory(node.GetName())
            structure = ProtectedStructure(node, category, DEFAULT_SAFE_DISTANCES[category], DEFAULT_STRUCTURE_RADIUS)
            self.setProtectedStructure(structure)
        return structure

    @staticmethod
    def removeProtectedStructure(node) -> None:
        """Stop protecting a structure (the node itself is kept)."""
        node.RemoveAttribute(STRUCTURE_ATTRIBUTE)

    def getProtectedStructures(self) -> list[ProtectedStructure]:
        """All structures to protect in the scene, in scene order."""
        structures = []
        for className in ("vtkMRMLModelNode", "vtkMRMLMarkupsCurveNode"):
            for node in slicer.util.getNodesByClass(className):
                structure = self.getProtectedStructure(node)
                if structure is not None:
                    structures.append(structure)
        return structures

    def sampleCutInBone(self, sheet: vtk.vtkPolyData, bone: vtk.vtkPolyData, spacing: float,
                        stops: Optional[list] = None, bounds: Optional[np.ndarray] = None) -> np.ndarray:
        """Points covering the part of a cutting sheet that lies inside the bone.

        Only this part cuts bone, so parts in air are ignored, as are parts beyond earlier
        lines of the osteotomy that this line stops at.

        :param sheet: cutting sheet.
        :param bone: closed bone mesh (inside test).
        :param spacing: largest distance between samples (mm).
        :param stops: (earlier sheet, side) pairs of a later osteotomy line.
        :param bounds: optional region (xmin, xmax, ...) to sample; default the bone's bounds.
        :return: (K, 3) points.
        """
        region = np.array(bone.GetBounds() if bounds is None else bounds, dtype=float)
        region += np.array([-1.0, 1.0] * 3) * spacing
        # Clip to the region by its six planes in turn: a plane's value is linear, so each clip is
        # exact (a box function is only evaluated at the corners of the large sheet triangles)
        clipped = sheet
        for axis in range(3):
            for side, value in ((1.0, region[2 * axis]), (-1.0, region[2 * axis + 1])):
                plane = vtk.vtkPlane()
                origin, normal = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
                origin[axis], normal[axis] = value, side
                plane.SetOrigin(origin)
                plane.SetNormal(normal)
                clip = vtk.vtkClipPolyData()
                clip.SetInputData(clipped)
                clip.SetClipFunction(plane)
                clip.Update()
                clipped = clip.GetOutput()
        pieces = list(self._triangleSamples(self._ensureTriangles(clipped), spacing))
        points = np.vstack(pieces) if pieces else np.zeros((0, 3))
        if len(points) and stops:
            points = points[~self.protectedByStops(points, stops)]
        if len(points) == 0:
            return points
        cloud = vtk.vtkPolyData()
        cloudPoints = vtk.vtkPoints()
        cloudPoints.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(points), deep=True))
        cloud.SetPoints(cloudPoints)
        enclosed = vtk.vtkSelectEnclosedPoints()
        enclosed.SetInputData(cloud)
        enclosed.SetSurfaceData(bone)
        enclosed.CheckSurfaceOff()
        enclosed.Update()
        inside = numpy_support.vtk_to_numpy(enclosed.GetOutput().GetPointData().GetArray("SelectedPoints")) > 0
        return points[inside]

    def _structureDistanceFunction(self, structure: ProtectedStructure) -> Callable[[np.ndarray], np.ndarray]:
        """Function giving the distance (mm) of points to the structure's surface: negative
        inside a closed model; for a curve, distance to the centreline minus the radius."""
        node = structure.node
        if node.IsA("vtkMRMLMarkupsCurveNode"):
            centre = np.array(slicer.util.arrayFromMarkupsCurvePoints(node, world=True), dtype=float)
            if len(centre) == 0:
                return lambda points: np.full(len(points), np.inf)
            dense = [centre[:1]]
            for start, end in zip(centre[:-1], centre[1:]):
                count = max(1, int(np.ceil(np.linalg.norm(end - start) / self.CENTRELINE_SPACING)))
                dense.append(start + np.outer(np.arange(1, count + 1) / count, end - start))
            dense = np.vstack(dense)
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(dense)
                return lambda points: tree.query(points)[0] - structure.radius
            except ImportError:
                locator = vtk.vtkStaticPointLocator()
                data = vtk.vtkPolyData()
                densePoints = vtk.vtkPoints()
                densePoints.SetData(numpy_support.numpy_to_vtk(dense, deep=True))
                data.SetPoints(densePoints)
                locator.SetDataSet(data)
                locator.BuildLocator()
                return lambda points: np.array([np.linalg.norm(dense[locator.FindClosestPoint(p)] - p)
                                                for p in points]) - structure.radius
        surface = self.mergeCoincidentPoints(self._ensureTriangles(self.getWorldPolyData(node)))
        closed = self.countOpenEdges(surface) == (0, 0)
        if closed:  # consistent outward normals give the sign
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(surface)
            normals.ConsistencyOn()
            normals.AutoOrientNormalsOn()
            normals.SplittingOff()
            normals.ComputeCellNormalsOn()
            normals.Update()
            surface = normals.GetOutput()
        implicitDistance = vtk.vtkImplicitPolyDataDistance()
        implicitDistance.SetInput(surface)

        def distances(points: np.ndarray) -> np.ndarray:
            values = vtk.vtkDoubleArray()
            implicitDistance.FunctionValue(numpy_support.numpy_to_vtk(np.ascontiguousarray(points), deep=True), values)
            result = numpy_support.vtk_to_numpy(values).copy()
            return result if closed else np.abs(result)

        return distances

    def checkClearances(self, lines: list[OsteotomyLine], bone: vtk.vtkPolyData,
                        structures: list[ProtectedStructure]) -> list[ClearanceResult]:
        """Smallest distance between the cut and each enabled structure to protect (headless).

        The cut is each line's sheet where it lies inside the bone (sampleCutInBone), minus
        half the blade width. It is sampled every CLEARANCE_COARSE_SPACING, then every
        CLEARANCE_FINE_SPACING near the closest samples.

        :param lines: the osteotomy's lines (sheet, options, stops).
        :param bone: closed bone mesh the cut uses (world coordinates).
        :param structures: structures to check; disabled ones are skipped.
        :return: one result per enabled structure: Safe (at least its safe distance), Too close,
            or Cut enters structure (clearance below 0).
        """
        coarse = []
        for line in lines:
            halfKerf = self.effectiveKerfWidth(line.options, bool(line.stops)) / 2.0
            coarse.append((line, halfKerf, self.sampleCutInBone(line.sheet, bone, self.CLEARANCE_COARSE_SPACING,
                                                                  line.stops)))
        boneBounds = np.array(bone.GetBounds())
        results = []
        for structure in (s for s in structures if s.enabled):
            distanceTo = self._structureDistanceFunction(structure)
            best, bestPoint, bestLine = np.inf, None, ""
            for line, halfKerf, samples in coarse:
                if len(samples) == 0:
                    continue
                values = distanceTo(samples)
                near = samples[values <= values.min() + 2.0 * self.CLEARANCE_COARSE_SPACING]
                region = np.empty(6)
                region[0::2] = np.maximum(near.min(axis=0) - self.CLEARANCE_COARSE_SPACING, boneBounds[0::2])
                region[1::2] = np.minimum(near.max(axis=0) + self.CLEARANCE_COARSE_SPACING, boneBounds[1::2])
                extents = np.sort(np.maximum(region[1::2] - region[0::2], 0.0))
                spacing = max(self.CLEARANCE_FINE_SPACING,
                              float(np.sqrt(extents[1] * extents[2] / self.CLEARANCE_MAX_FINE_SAMPLES)))
                fine = self.sampleCutInBone(line.sheet, bone, spacing, line.stops, region)
                candidates = np.vstack([fine, near]) if len(fine) else near
                values = distanceTo(candidates) - halfKerf
                index = int(np.argmin(values))
                if values[index] < best:
                    best, bestPoint, bestLine = float(values[index]), candidates[index], line.curve.GetName()
            if best < 0:
                status = ClearanceStatus.ENTERS
            elif best < structure.safeDistance:
                status = ClearanceStatus.TOO_CLOSE
            else:
                status = ClearanceStatus.SAFE
            results.append(ClearanceResult(structure, best, bestPoint, bestLine, status))
        return results

    def getBoneForChecks(self, parameterNode: OsteotomyCutsParameterNode, buildSolid: bool) -> vtk.vtkPolyData:
        """The bone mesh for clearance checks: the one the cut uses (the solid copy with "Treat
        bone as solid"), or the model itself if the solid copy is not built yet and buildSolid
        is False (live preview)."""
        inputModel = parameterNode.inputModel
        if (parameterNode.treatBoneAsSolid and not self.isSolidModel(inputModel) and importNdimage() is not None
                and (buildSolid or inputModel.GetID() in self._solidCache)):
            return self.getModelToCut(parameterNode)[0]
        return self.getWorldPolyData(inputModel)

    def checkClearancesForParameters(self, parameterNode: OsteotomyCutsParameterNode,
                                     buildSolid: bool = True) -> list[ClearanceResult]:
        """checkClearances for the selected osteotomy and all structures in the scene (the
        structure being cut itself is skipped). Empty if there are no structures.

        :raises ValueError: if the inputs are invalid.
        """
        structures = [s for s in self.getProtectedStructures() if s.enabled
                      and s.node.GetID() != parameterNode.inputModel.GetID()]
        if not structures:
            return []
        lines = self.buildOsteotomy(parameterNode)
        return self.checkClearances(lines, self.getBoneForChecks(parameterNode, buildSolid), structures)

    def updateClearanceDisplay(self, parameterNode: OsteotomyCutsParameterNode,
                               results: list[ClearanceResult]) -> None:
        """Mark the closest point of the cut to each structure ("ClosestToCut" points, locked,
        labelled with the distance) and colour the sheet preview by the worst status."""
        markups = parameterNode.clearanceMarkups
        if markups is None:
            markups = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "ClosestToCut")
            markups.SetHideFromEditors(True)
            markups.SetLocked(True)
            markups.CreateDefaultDisplayNodes()
            markups.GetDisplayNode().SetPointLabelsVisibility(True)
            parameterNode.clearanceMarkups = markups
        wasModifying = markups.StartModify()
        markups.RemoveAllControlPoints()
        worst = None
        order = [ClearanceStatus.SAFE, ClearanceStatus.TOO_CLOSE, ClearanceStatus.ENTERS]
        for result in results:
            if result.closestPoint is None:
                continue
            index = markups.AddControlPointWorld(vtk.vtkVector3d(*result.closestPoint))
            markups.SetNthControlPointLabel(index, f"{result.structure.node.GetName()}: {result.clearance:.1f} mm")
            markups.SetNthControlPointLocked(index, True)
            if worst is None or order.index(result.status) > order.index(worst):
                worst = result.status
        markups.EndModify(wasModifying)
        colour = self.STATUS_COLOURS[worst] if worst is not None else (0.9, 0.2, 0.2)
        markups.GetDisplayNode().SetSelectedColor(colour)
        markups.GetDisplayNode().SetVisibility(bool(results))
        sheetNode = parameterNode.sheetModel
        if sheetNode is not None and sheetNode.GetDisplayNode() is not None:
            sheetNode.GetDisplayNode().SetColor(colour)

    @staticmethod
    def clearanceSummary(results: list[ClearanceResult]) -> list[dict]:
        """Clearance results as plain values (for the SafetyResults attribute of segments)."""
        return [{"structure": result.structure.node.GetName(), "category": result.structure.category.value,
                 "safeDistance_mm": result.structure.safeDistance,
                 "clearance_mm": None if not np.isfinite(result.clearance) else round(result.clearance, 2),
                 "status": result.status.value, "line": result.lineName} for result in results]

    def recordSafetyResults(self, nodes: list[vtkMRMLModelNode], results: list[ClearanceResult],
                            override: bool) -> None:
        """Store the clearance results on the bone segments (OsteotomyCuts.SafetyResults, JSON)
        and, if the surgeon cut despite a warning, OsteotomyCuts.SafetyOverride = "true"."""
        text = json.dumps(self.clearanceSummary(results))
        for node in nodes:
            node.SetAttribute("OsteotomyCuts.SafetyResults", text)
            if override:
                node.SetAttribute("OsteotomyCuts.SafetyOverride", "true")

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
        self.assertEqual(parameterNode.options.endExtension, 0.0)
        self.assertTrue(parameterNode.options.capCutFaces)
        self.assertEqual(parameterNode.options.refineEdgeLength, 0.0)
        self.assertTrue(parameterNode.treatBoneAsSolid)
        self.assertEqual(parameterNode.solidVoxelSize, 0.25)
        self.assertEqual(parameterNode.solidGapSeal, 1.5)
        self.assertEqual(parameterNode.minSegmentPercent, 1.0)

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
        """Cut with one sheet per path, all extruded along the same direction (and options.depth)."""
        depth = options.depth if options is not None and options.depth > 0 else None
        sheets = self._sheets(polyData, paths, direction, closed, [depth] * len(paths))
        return OsteotomyCutsLogic().cutPolyData(polyData, sheets, options or CutOptions())

    @staticmethod
    def _sheets(polyData, paths, direction, closed=False, depths=None) -> list:
        """One sheet per path, extruded along the direction; depths: one per path (None = through)."""
        logic = OsteotomyCutsLogic()
        extent = logic.computeAutoExtent(polyData)
        return [logic.buildSheetPolyData(np.array(path, dtype=float), np.array(direction, dtype=float),
                                         extent, closed, depth=depth)
                for path, depth in zip(paths, depths or [None] * len(paths))]

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
        """A closed circle extruded through a sphere: uncapped, top cap, bottom cap and band;
        capped, the solid core (joined through the cut face) and the ring around it."""
        radius, circleRadius = 30.0, 10.0
        angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        z = np.sqrt(radius ** 2 - circleRadius ** 2)
        circle = np.column_stack([circleRadius * np.cos(angles), circleRadius * np.sin(angles),
                                  np.full_like(angles, z)])
        options = CutOptions()
        options.capCutFaces = False
        fragments = self._cut(self._sphere(radius), [circle], [0.0, 0.0, 1.0], closed=True, options=options)

        self.assertEqual(len(fragments), 3)
        band = fragments[0]
        caps = sorted(fragments[1:], key=lambda f: f.GetBounds()[4])
        self.assertLess(caps[0].GetBounds()[5], 0.0)  # bottom cap
        self.assertGreater(caps[1].GetBounds()[4], 0.0)  # top cap
        self.assertGreater(band.GetBounds()[1], circleRadius)

        fragments = self._cut(self._sphere(radius), [circle], [0.0, 0.0, 1.0], closed=True)
        self.assertEqual(len(fragments), 2)
        core = min(fragments, key=lambda f: f.GetBounds()[1])
        np.testing.assert_allclose(core.GetBounds()[4:], (-radius, radius), atol=0.1)
        self.assertLessEqual(core.GetBounds()[1], circleRadius + 1e-6)

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
    def _configure(model, curve, direction=DOWN, solid=False) -> OsteotomyCutsParameterNode:
        """Parameters for cutting model with curve; the model is cut as it is unless solid is set."""
        parameterNode = OsteotomyCutsLogic().getParameterNode()
        parameterNode.treatBoneAsSolid = solid
        parameterNode.minSegmentPercent = 0.0  # every piece kept, unless a test joins them
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
        parameterNode = self._configure(model, curve, direction=(0.0, 0.0, 1.0))
        parameterNode.options.capCutFaces = False  # three open pieces
        band, cap1, cap2 = logic.applyCut(parameterNode)
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
        parameterNode.treatBoneAsSolid = False
        fragment = logic.applyCut(parameterNode)[0]
        parameterNode.inputModel = fragment
        self.assertIn("produced by this osteotomy", logic.validateInputs(parameterNode))
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
        """The preview sheet follows the depth; a groove gives one fragment, with the ideal blade
        (a thin numerical layer) or a kerf."""
        logic = OsteotomyCutsLogic()
        model = self._addModel(self._box(), "Box")
        curve = self._addCurve(self.X_CUT_PATH, "CutA")
        parameterNode = self._configure(model, curve)

        parameterNode.options.depth = 12.0
        self.assertIsNone(logic.validateInputs(parameterNode))
        sheetNode = logic.updateSheetModel(parameterNode)
        self.assertAlmostEqual(sheetNode.GetPolyData().GetBounds()[4], 50.0 - 12.0, places=6)
        nodes = logic.applyCut(parameterNode)  # ideal blade: a thin groove, nothing is separated
        self.assertEqual(len(nodes), 1)
        points = self._points(nodes[0].GetPolyData())
        halfLayer = logic.LIMITED_IDEAL_KERF_WIDTH / 2.0
        inGroove = (np.abs(points[:, 0] - 1.3) < halfLayer - 0.001) & (points[:, 2] > 50.0 - 12.0)
        self.assertFalse(np.any(inGroove))

        parameterNode.options.kerfWidth = 1.0
        self.assertIsNone(logic.validateInputs(parameterNode))
        nodes = logic.applyCut(parameterNode)  # a groove: nothing is separated
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].GetName(), "Box_CutA_1")
        self.assertFalse(self._isVisible(model))
        points = self._points(nodes[0].GetPolyData())
        inGroove = (np.abs(points[:, 0] - 1.3) < 0.5 - 0.01) & (points[:, 2] > 50.0 - 12.0)
        self.assertFalse(np.any(inGroove))

    def test_oldSceneUpgrade(self):
        """A scene saved before newer options existed loads, keeps its values and gets defaults."""
        import os
        logic = OsteotomyCutsLogic()
        parameterNode = self._configure(self._addModel(self._box(), "Box"), self._addCurve(self.X_CUT_PATH, "CutA"))
        parameterNode.options.extension = 250.0
        rawNode = parameterNode.parameterNode
        del parameterNode  # the wrapper would try (and fail) to re-read the node as it is edited
        newOptions = ("options.kerfWidth", "options.depth", "options.capCutFaces", "options.refineEdgeLength",
                      "options.endExtension", "treatBoneAsSolid", "solidVoxelSize", "solidGapSeal",
                      "minSegmentPercent")
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
        self.assertTrue(parameterNode.treatBoneAsSolid)
        self.assertEqual(parameterNode.solidVoxelSize, 0.25)
        self.assertEqual(parameterNode.solidGapSeal, 1.5)
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
        options.capCutFaces = False  # cap triangles span the cut face
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

    #
    # Phase 2 step 3: kerf removal
    #

    @staticmethod
    def _kerfOptions(kerfWidth: float, depth: float = 0.0) -> CutOptions:
        options = CutOptions()
        options.kerfWidth = kerfWidth
        options.depth = depth
        return options

    def test_kerf_throughCut(self):
        """A through-cut with a kerf removes a slot of exactly the kerf width."""
        x, kerf = 1.3, 2.0
        box = self._box()
        fragments = self._cut(box, [self.X_CUT_PATH], self.DOWN, options=self._kerfOptions(kerf))

        self.assertEqual(len(fragments), 2)
        boundsList = sorted((fragment.GetBounds() for fragment in fragments), key=lambda b: b[0])
        self.assertAlmostEqual(boundsList[0][1], x - kerf / 2.0, places=6)
        self.assertAlmostEqual(boundsList[1][0], x + kerf / 2.0, places=6)
        for bounds in boundsList:
            np.testing.assert_allclose(bounds[2:], (-50.0, 50.0, -50.0, 50.0), atol=1e-6)
        for fragment in fragments:
            names = {fragment.GetPointData().GetArrayName(i)
                     for i in range(fragment.GetPointData().GetNumberOfArrays())}
            self.assertFalse(any(name.startswith(SHEET_SIDE_PREFIX) for name in names))
            self.assertNotIn("SheetDistance", names)
        self.assertIsNone(box.GetPointData().GetArray("SheetDistance"))

    def test_kerf_groove(self):
        """A depth-limited kerf cut removes a groove with a rounded bottom and leaves one piece."""
        x, kerf, depth = 1.3, 2.0, 10.0
        options = self._kerfOptions(kerf, depth)
        options.capCutFaces = False
        fragments = self._cut(self._box(), [self.X_CUT_PATH], self.DOWN, options=options)

        self.assertEqual(len(fragments), 1)
        points = self._points(fragments[0])
        bottom = 50.0 - depth  # inner edge of the sheet
        # 0.01 mm: the groove floor is curved, the mesh between rim points is not
        self.assertFalse(np.any((np.abs(points[:, 0] - x) < kerf / 2.0 - 0.01) & (points[:, 2] > bottom)))
        # Without capping the groove is open; the deepest point of its rounded floor, on the
        # open boundary, is half the kerf below the sheet edge
        edges = vtk.vtkFeatureEdges()
        edges.SetInputData(fragments[0])
        edges.BoundaryEdgesOn()
        edges.NonManifoldEdgesOff()
        edges.FeatureEdgesOff()
        edges.ManifoldEdgesOff()
        edges.Update()
        floorDepth = self._points(edges.GetOutput())[:, 2].min()
        self.assertGreater(floorDepth, bottom - kerf / 2.0 - 0.01)
        self.assertAlmostEqual(floorDepth, bottom - kerf / 2.0, delta=0.2)
        # Groove walls reach the top face on both sides
        top = points[:, 2] > 50.0 - 1e-6
        self.assertAlmostEqual(points[top & (points[:, 0] < x), 0].max(), x - kerf / 2.0, places=6)
        self.assertAlmostEqual(points[top & (points[:, 0] > x), 0].min(), x + kerf / 2.0, places=6)

    def test_kerf_enclosedShell(self):
        """With a kerf, each half of an inner shell stays with its side's fragment."""
        x, kerf = 1.3, 1.0
        mesh = self._append(self._sphere(30.0), self._sphere(10.0))
        fragments = self._cut(mesh, [[[x, -20.0, 29.0], [x, 20.0, 29.0]]], self.DOWN,
                              options=self._kerfOptions(kerf))

        self.assertEqual(len(fragments), 2)
        for fragment in fragments:
            radii = self._radii(fragment)
            self.assertTrue(np.any(np.abs(radii - 30.0) < 0.5), "outer shell part missing")
            self.assertTrue(np.any(np.abs(radii - 10.0) < 0.5), "inner shell part missing")
            xs = self._points(fragment)[:, 0]
            self.assertTrue(np.all(xs <= x - kerf / 2.0 + 1e-3) or np.all(xs >= x + kerf / 2.0 - 1e-3))

    def test_kerf_multipleSheets(self):
        """Two crossing kerf sheets cut a box into four quadrants."""
        x, y, kerf = 1.3, 2.7, 1.0
        fragments = self._cut(self._box(), [[[x, -40.0, 50.0], [x, 40.0, 50.0]],
                                            [[-40.0, y, 50.0], [40.0, y, 50.0]]], self.DOWN,
                              options=self._kerfOptions(kerf))

        self.assertEqual(len(fragments), 4)
        quadrants = set()
        for fragment in fragments:
            points = self._points(fragment)
            self.assertTrue(np.all(np.abs(points[:, 0] - x) >= kerf / 2.0 - 1e-6))
            self.assertTrue(np.all(np.abs(points[:, 1] - y) >= kerf / 2.0 - 1e-6))
            centre = points.mean(axis=0)
            quadrants.add((bool(centre[0] > x), bool(centre[1] > y)))
        self.assertEqual(len(quadrants), 4)

    def test_kerf_missesModel(self):
        """A kerf cut whose sheet does not reach the model raises ValueError."""
        path = [[1.3, -40.0, 70.0], [1.3, 40.0, 70.0]]  # 20 mm above the box, 10 mm deep
        with self.assertRaises(ValueError):
            self._cut(self._box(), [path], self.DOWN, options=self._kerfOptions(1.0, 10.0))
        with self.assertRaises(ValueError):
            OsteotomyCutsLogic().removeKerf(self._box(), vtk.vtkPolyData(), 0.0)

    #
    # Phase 2 step 4: capping through-cuts
    #

    def _assertWatertight(self, fragment: vtk.vtkPolyData) -> None:
        """Closed and crack-free once the duplicate points of the display normals are joined."""
        self.assertEqual(self._openEdgeCount(OsteotomyCutsLogic().mergeCoincidentPoints(fragment)), 0)

    def _assertVolumes(self, fragments, expected, relativeTolerance):
        volumes = sorted(self._volume(fragment) for fragment in fragments)
        np.testing.assert_allclose(volumes, sorted(expected), rtol=relativeTolerance)

    def test_cap_planar(self):
        """Zero-kerf and kerf through-cuts of a box give watertight fragments of the right volume."""
        x = 1.3
        fragments = self._cut(self._box(), [self.X_CUT_PATH], self.DOWN)
        self.assertEqual(len(fragments), 2)
        for fragment in fragments:
            self._assertWatertight(fragment)
            self.assertIsNotNone(fragment.GetPointData().GetNormals())
        self._assertVolumes(fragments, [(50.0 + x) * 1e4, (50.0 - x) * 1e4], 1e-6)

        kerf = 2.0
        fragments = self._cut(self._box(), [self.X_CUT_PATH], self.DOWN, options=self._kerfOptions(kerf))
        self.assertEqual(len(fragments), 2)
        for fragment in fragments:
            self._assertWatertight(fragment)
        self._assertVolumes(fragments, [(50.0 + x - kerf / 2.0) * 1e4, (50.0 - x - kerf / 2.0) * 1e4], 1e-6)

    def test_cap_Lshape(self):
        """A folded sheet with a kerf: sharp inner corner, rounded outer corner, both watertight."""
        c, kerf = 20.3, 1.0
        r = kerf / 2.0
        fragments = self._cut(self._box(), [[[c, -40.0, 50.0], [c, c, 50.0], [40.0, c, 50.0]]], self.DOWN,
                              options=self._kerfOptions(kerf))

        self.assertEqual(len(fragments), 2)
        for fragment in fragments:
            self._assertWatertight(fragment)
        corner = (50.0 - c - r) * (c - r + 50.0) * 100.0
        grownCorner = (50.0 - c + r) * (c + r + 50.0) - r ** 2 + np.pi * r ** 2 / 4.0  # rounded outer corner
        self._assertVolumes(fragments, [corner, (1e4 - grownCorner) * 100.0], 1e-3)

    def test_cap_enclosedShell(self):
        """An inner void shell becomes a hole in each cap; each fragment is one watertight solid."""
        x, kerf = 1.3, 1.0
        inner = vtk.vtkReverseSense()  # a void: its surface faces inwards, as from a segmentation
        inner.SetInputData(self._sphere(10.0))
        inner.ReverseCellsOn()
        inner.ReverseNormalsOn()
        inner.Update()
        fragments = self._cut(self._append(self._sphere(30.0), inner.GetOutput()),
                              [[[x, -20.0, 29.0], [x, 20.0, 29.0]]], self.DOWN, options=self._kerfOptions(kerf))

        self.assertEqual(len(fragments), 2)

        def capVolume(radius: float, planeX: float) -> float:
            height = radius - planeX  # of the part x > planeX
            return np.pi * height ** 2 * (3.0 * radius - height) / 3.0

        expected = []
        for planeX in (x + kerf / 2.0, -(x - kerf / 2.0)):  # x > x + r, and x < x - r mirrored
            expected.append(capVolume(30.0, planeX) - capVolume(10.0, planeX))
        for fragment in fragments:
            self._assertWatertight(fragment)
            radii = self._radii(fragment)
            self.assertTrue(np.any(np.abs(radii - 10.0) < 0.5), "inner shell part missing")
            regions = vtk.vtkPolyDataConnectivityFilter()
            regions.SetInputData(OsteotomyCutsLogic().mergeCoincidentPoints(fragment))
            regions.SetExtractionModeToAllRegions()
            regions.Update()
            self.assertEqual(regions.GetNumberOfExtractedRegions(), 1)  # joined by the cap
        self._assertVolumes(fragments, expected, 0.01)  # sphere tessellation

    def test_cap_closedCurve(self):
        """A closed sheet (cylinder) is capped on its curved face; the volumes add up."""
        radius, circleRadius = 30.0, 10.0
        angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        z = np.sqrt(radius ** 2 - circleRadius ** 2)
        circle = np.column_stack([circleRadius * np.cos(angles), circleRadius * np.sin(angles),
                                  np.full_like(angles, z)])
        sphere = self._sphere(radius)
        fragments = self._cut(sphere, [circle], [0.0, 0.0, 1.0], closed=True)

        self.assertEqual(len(fragments), 2)
        for fragment in fragments:
            self._assertWatertight(fragment)
        total = sum(self._volume(fragment) for fragment in fragments)
        self.assertAlmostEqual(total, self._volume(sphere), delta=1e-3 * self._volume(sphere))
        # The cap follows the curved sheet: no cap point far inside the 24-gon
        core = min(fragments, key=lambda f: f.GetBounds()[1])
        polygonInradius = circleRadius * np.cos(np.pi / 24)
        coreRadii = np.linalg.norm(self._points(core)[:, :2], axis=1)
        insideSphere = np.abs(self._points(core)[:, 2]) < z - 1.0
        self.assertGreater(coreRadii[insideSphere].min(), polygonInradius - self.CAP_CHORD_TOLERANCE)

    CAP_CHORD_TOLERANCE = 0.05

    def test_cap_multipleSheets(self):
        """Two crossing kerf sheets give four watertight quadrants of the right volume."""
        x, y, kerf = 1.3, 2.7, 1.0
        r = kerf / 2.0
        fragments = self._cut(self._box(), [[[x, -40.0, 50.0], [x, 40.0, 50.0]],
                                            [[-40.0, y, 50.0], [40.0, y, 50.0]]], self.DOWN,
                              options=self._kerfOptions(kerf))

        self.assertEqual(len(fragments), 4)
        for fragment in fragments:
            self._assertWatertight(fragment)
        widthsX, widthsY = (50.0 + x - r, 50.0 - x - r), (50.0 + y - r, 50.0 - y - r)
        self._assertVolumes(fragments, [wx * wy * 100.0 for wx in widthsX for wy in widthsY], 1e-6)

    def test_cap_recut(self):
        """A capped fragment (duplicate points at sharp edges) can be cut again."""
        fragments = self._cut(self._box(), [self.X_CUT_PATH], self.DOWN)
        half = fragments[0]
        self.assertGreater(half.GetNumberOfPoints(),
                           OsteotomyCutsLogic().mergeCoincidentPoints(half).GetNumberOfPoints())
        pieces = self._cut(half, [[[-40.0, 2.7, 50.0], [40.0, 2.7, 50.0]]], self.DOWN)

        self.assertEqual(len(pieces), 2)
        for piece in pieces:
            self._assertWatertight(piece)
        self.assertAlmostEqual(sum(self._volume(piece) for piece in pieces), self._volume(half), places=3)

    def test_cap_disabled(self):
        """Without capping, the cut faces stay open."""
        options = CutOptions()
        options.capCutFaces = False
        for fragment in self._cut(self._box(), [self.X_CUT_PATH], self.DOWN, options=options):
            self.assertGreater(self._openEdgeCount(fragment), 0)
        options = self._kerfOptions(1.0, 10.0)
        options.capCutFaces = False
        groove = self._cut(self._box(), [self.X_CUT_PATH], self.DOWN, options=options)
        self.assertEqual(len(groove), 1)
        self.assertGreater(self._openEdgeCount(groove[0]), 0)

    #
    # Phase 2 step 5: capping grooves and combined cuts
    #

    def _assertOnCutSurfaces(self, fragment, sheets, kerf, modelSurface, tolerance=None) -> None:
        """No bone is left inside any kerf, and every point off the model surface (cap points)
        lies on the cut surface of one of the sheets.

        :param modelSurface: function (points) -> bool array, True for points on the uncut model.
        """
        tolerance = OsteotomyCutsLogic.CAP_TOLERANCE if tolerance is None else tolerance
        logic = OsteotomyCutsLogic()
        points = self._points(fragment)
        excess = np.stack([np.abs(numpy_support.vtk_to_numpy(logic.computeSheetDistance(fragment, sheet)
                                                             .GetPointData().GetArray("SheetDistance")))
                           - kerf / 2.0 for sheet in sheets])
        self.assertGreater(excess.min(), -tolerance, "bone left inside the kerf")
        offModel = ~modelSurface(points)
        self.assertTrue(np.any(offModel))
        self.assertLess(np.abs(excess[:, offModel]).min(axis=0).max(), tolerance, "cap point off the cut surface")

    @staticmethod
    def _onBoxFaces(points: np.ndarray, halfSize: float = 50.0) -> np.ndarray:
        return np.abs(points).max(axis=1) > halfSize - 1e-6

    def _removedVolume(self, fragments, model) -> float:
        return self._volume(model) - sum(self._volume(fragment) for fragment in fragments)

    def test_cap_groove(self):
        """A straight groove is lined along its walls and rounded floor: one watertight solid
        with exactly the groove removed."""
        x, kerf, depth = 1.3, 2.0, 10.0
        r = kerf / 2.0
        box = self._box()
        fragments = self._cut(box, [self.X_CUT_PATH], self.DOWN, options=self._kerfOptions(kerf, depth))

        self.assertEqual(len(fragments), 1)
        self._assertWatertight(fragments[0])
        sheets = self._sheets(box, [self.X_CUT_PATH], self.DOWN, depths=[depth])
        self._assertOnCutSurfaces(fragments[0], sheets, kerf, self._onBoxFaces)
        # Slot of the kerf width down to the sheet edge, and a half cylinder below it, 100 mm long
        expected = 100.0 * (kerf * depth + np.pi * r ** 2 / 2.0)
        self.assertAlmostEqual(self._removedVolume(fragments, box), expected, delta=0.002 * expected)
        # The floor reaches half the kerf below the sheet edge
        points = self._points(fragments[0])
        inGroove = (np.abs(points[:, 0] - x) < r - 1e-3) & ~self._onBoxFaces(points)
        self.assertAlmostEqual(points[inGroove, 2].min(), 50.0 - depth - r, delta=OsteotomyCutsLogic.CAP_TOLERANCE)

    def test_cap_grooveLshape(self):
        """A folded groove: rounded outer corner (a quarter sphere on the floor) and sharp inner
        corner, lined watertight, with the removed volume of the swept kerf."""
        c, kerf, depth = 20.3, 1.0, 10.0
        r = kerf / 2.0
        box = self._box()
        path = [[c, -40.0, 50.0], [c, c, 50.0], [40.0, c, 50.0]]
        fragments = self._cut(box, [path], self.DOWN, options=self._kerfOptions(kerf, depth))

        self.assertEqual(len(fragments), 1)
        self._assertWatertight(fragments[0])
        self._assertOnCutSurfaces(fragments[0], self._sheets(box, [path], self.DOWN, depths=[depth]), kerf,
                                  self._onBoxFaces)
        # Plan view: strips of width 2r along 100 mm of path, plus the rounded outer corner, less
        # the overlap at the inner corner. Floor: half a tube around the bent sheet edge.
        length = 100.0
        slot = depth * (2.0 * r * length + r ** 2 * (np.pi / 4.0 - 1.0))
        floor = (np.pi * r ** 2 * length + np.pi * r ** 3 / 3.0 - 4.0 * r ** 3 / 3.0) / 2.0
        self.assertAlmostEqual(self._removedVolume(fragments, box), slot + floor, delta=0.002 * (slot + floor))

    def test_cap_grooveSphere(self):
        """A groove into a curved surface: its rim is one loop over both walls and the floor."""
        kerf, depth = 1.0, 5.0
        sphere = self._sphere(30.0)
        path = [[1.3, -40.0, 29.0], [1.3, 40.0, 29.0]]
        fragments = self._cut(sphere, [path], self.DOWN, options=self._kerfOptions(kerf, depth))

        self.assertEqual(len(fragments), 1)
        self._assertWatertight(fragments[0])
        onSphere = lambda points: np.abs(np.linalg.norm(points, axis=1) - 30.0) < 0.1  # tessellated sphere
        self._assertOnCutSurfaces(fragments[0], self._sheets(sphere, [path], self.DOWN, depths=[depth]), kerf,
                                  onSphere)
        self.assertGreater(self._removedVolume(fragments, sphere), 0.0)

    def test_cap_grooveClosedCurve(self):
        """A ring groove from a closed curve: the cut surface is unrolled to an annulus whose
        inner and outer walls have different lengths."""
        radius, circleRadius, kerf, depth = 30.0, 10.0, 1.0, 5.0
        angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        z = np.sqrt(radius ** 2 - circleRadius ** 2)
        circle = np.column_stack([circleRadius * np.cos(angles), circleRadius * np.sin(angles),
                                  np.full_like(angles, z)])
        sphere = self._sphere(radius)
        fragments = self._cut(sphere, [circle], self.DOWN, closed=True, options=self._kerfOptions(kerf, depth))

        self.assertEqual(len(fragments), 1)
        self._assertWatertight(fragments[0])
        onSphere = lambda points: np.abs(np.linalg.norm(points, axis=1) - radius) < 0.1
        sheets = self._sheets(sphere, [circle], self.DOWN, closed=True, depths=[depth])
        self._assertOnCutSurfaces(fragments[0], sheets, kerf, onSphere)

    def test_cap_crossingGrooves(self):
        """Two crossing grooves of different depths: the second is lined across the lining of
        the first."""
        x, y, kerf = 1.3, 2.7, 1.0
        paths = [[[x, -40.0, 50.0], [x, 40.0, 50.0]], [[-40.0, y, 50.0], [40.0, y, 50.0]]]
        depths = [10.0, 6.0]
        box = self._box()
        sheets = self._sheets(box, paths, self.DOWN, depths=depths)
        fragments = OsteotomyCutsLogic().cutPolyData(box, sheets, self._kerfOptions(kerf))

        self.assertEqual(len(fragments), 1)
        self._assertWatertight(fragments[0])
        self._assertOnCutSurfaces(fragments[0], sheets, kerf, self._onBoxFaces)
        r = kerf / 2.0
        single = [100.0 * (kerf * depth + np.pi * r ** 2 / 2.0) for depth in depths]
        removed = self._removedVolume(fragments, box)
        self.assertGreater(removed, max(single))
        self.assertLess(removed, sum(single))

    def test_cap_fallbackTriangulation(self):
        """Without the Delaunay triangulation (e.g. no scipy), caps are still closed: the loops
        are triangulated alone and the interior points inserted."""
        kerf, depth = 1.0, 10.0
        box = self._box()
        logic = OsteotomyCutsLogic()
        logic._delaunayInLoops = lambda planar, edges: None
        sheets = self._sheets(box, [self.X_CUT_PATH], self.DOWN, depths=[depth])
        fragments = logic.cutPolyData(box, sheets, self._kerfOptions(kerf, depth))

        self.assertEqual(len(fragments), 1)
        self._assertWatertight(fragments[0])
        self._assertOnCutSurfaces(fragments[0], sheets, kerf, self._onBoxFaces)

    def test_cap_recoverEdges(self):
        """A missing rim edge is recovered by flips (no point added, triangles counter-clockwise)."""
        points = np.array([[0.0, 0.0], [2.0, -0.2], [4.0, 0.0], [2.0, 0.2]])
        triangles = np.array([[0, 1, 3], [1, 2, 3]])  # Delaunay-like: diagonal 1-3, not 0-2
        recovered = OsteotomyCutsLogic._recoverEdges(points, triangles, np.array([[0, 2]]))
        self.assertIsNotNone(recovered)
        edges = {tuple(sorted((int(t[k]), int(t[(k + 1) % 3])))) for t in recovered for k in range(3)}
        self.assertIn((0, 2), edges)
        corners = points[recovered]
        area = ((corners[:, 1, 0] - corners[:, 0, 0]) * (corners[:, 2, 1] - corners[:, 0, 1])
                - (corners[:, 1, 1] - corners[:, 0, 1]) * (corners[:, 2, 0] - corners[:, 0, 0]))
        self.assertTrue(np.all(area > 0))
        self.assertAlmostEqual(float(area.sum()) / 2.0, 0.8)  # same area as before
        # A point on the edge makes it unrecoverable
        onEdge = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [2.0, 1.0], [2.0, -1.0]])
        fan = np.array([[0, 4, 1], [1, 4, 2], [0, 1, 3], [1, 2, 3]])
        self.assertIsNone(OsteotomyCutsLogic._recoverEdges(onEdge, fan, np.array([[0, 2]])))

    def test_cap_refinementBudget(self):
        """When a cap's refinement budget runs out, the cap stays closed (coarser)."""
        logic = OsteotomyCutsLogic()
        logic.CAP_MIN_POINT_BUDGET = 0
        logic.CAP_POINT_BUDGET_PER_RIM_POINT = 0
        sheets = self._sheets(self._sphere(30.0), [[[1.3, -40.0, 29.0], [1.3, 40.0, 29.0]]], self.DOWN, depths=[5.0])
        fragments = logic.cutPolyData(self._sphere(30.0), sheets, self._kerfOptions(1.0, 5.0))
        self.assertEqual(len(fragments), 1)
        self._assertWatertight(fragments[0])

    def test_cap_grooveAndThroughCut(self):
        """A groove, then a through-cut across it: two watertight fragments, each lined."""
        x, y, kerf = 1.3, 2.7, 1.0
        paths = [[[-40.0, y, 50.0], [40.0, y, 50.0]], [[x, -40.0, 50.0], [x, 40.0, 50.0]]]
        box = self._box()
        sheets = self._sheets(box, paths, self.DOWN, depths=[10.0, None])
        fragments = OsteotomyCutsLogic().cutPolyData(box, sheets, self._kerfOptions(kerf))

        self.assertEqual(len(fragments), 2)
        for fragment in fragments:
            self._assertWatertight(fragment)
            self._assertOnCutSurfaces(fragment, sheets, kerf, self._onBoxFaces)

    #
    # Release part 0b: limiting the cut
    #

    @staticmethod
    def _lineLength(lines: vtk.vtkPolyData) -> float:
        points = numpy_support.vtk_to_numpy(lines.GetPoints().GetData())
        pairs = numpy_support.vtk_to_numpy(lines.GetLines().GetConnectivityArray()).reshape(-1, 2)
        return float(np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1).sum())

    def test_sheet_endExtension(self):
        """The sheet continues past the path ends by endExtension; the outward reach stays the extent."""
        logic = OsteotomyCutsLogic()
        path = np.array([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        down = np.array([0.0, 0.0, -1.0])
        points = self._points(logic.buildSheetPolyData(path, down, 50.0, depth=8.0, endExtension=5.0))
        np.testing.assert_allclose(points[0::2, 0], [-15.0, -10.0, 10.0, 15.0])
        np.testing.assert_allclose(points[0::2, 2], 50.0)
        np.testing.assert_allclose(points[1::2, 2], -8.0)
        # By default the ends reach as far as the extent
        np.testing.assert_allclose(self._points(logic.buildSheetPolyData(path, down, 50.0))[0::2, 0],
                                   [-60.0, -10.0, 10.0, 60.0])
        for endExtension in (0.0, -1.0):
            with self.assertRaises(ValueError, msg=str(endExtension)):
                logic.buildSheetPolyData(path, down, 50.0, endExtension=endExtension)

    def test_limitedIdealCut(self):
        """A limited cut with the ideal blade ends where its sheet ends: it removes a thin groove
        and leaves one watertight piece (a zero-width cut would split the box in two)."""
        logic = OsteotomyCutsLogic()
        options = self._kerfOptions(0.0, 10.0)
        width = logic.LIMITED_IDEAL_KERF_WIDTH
        self.assertTrue(logic.isLimitedCut(options))
        self.assertEqual(logic.effectiveKerfWidth(options), width)
        self.assertEqual(logic.effectiveKerfWidth(self._kerfOptions(0.0)), 0.0)
        self.assertEqual(logic.effectiveKerfWidth(self._kerfOptions(1.0, 10.0)), 1.0)

        box = self._box()
        fragments = self._cut(box, [self.X_CUT_PATH], self.DOWN, options=options)
        self.assertEqual(len(fragments), 1)
        self._assertWatertight(fragments[0])
        expected = 100.0 * (width * 10.0 + np.pi * (width / 2.0) ** 2 / 2.0)
        self.assertAlmostEqual(self._volume(box) - self._volume(fragments[0]), expected, delta=0.01 * expected)

    def test_cap_fineRefinement(self):
        """Rim edges much shorter than the model scale (a fine refinement length) are not
        collapsed away: the groove is still capped, watertight and of the right volume."""
        options = self._kerfOptions(0.5, 10.0)
        options.refineEdgeLength = 0.02
        box = self._box()
        fragments = self._cut(box, [self.X_CUT_PATH], self.DOWN, options=options)
        self.assertEqual(len(fragments), 1)
        self._assertWatertight(fragments[0])
        expected = 100.0 * (0.5 * 10.0 + np.pi * 0.25 ** 2 / 2.0)
        self.assertAlmostEqual(self._volume(box) - self._volume(fragments[0]), expected, delta=0.01 * expected)

    def test_endExtension_sparesOtherBone(self):
        """Stopping the cut past the line ends spares bone beyond them (e.g. the skull base
        beyond a Le Fort I cut); an automatic reach cuts it too."""
        logic = OsteotomyCutsLogic()
        other = vtk.vtkTransformPolyDataFilter()
        other.SetInputData(self._box(20.0, 8))
        shift = vtk.vtkTransform()
        shift.Translate(0.0, 120.0, 0.0)
        other.SetTransform(shift)
        other.Update()
        model = self._append(self._box(), other.GetOutput())
        path = np.array(self.X_CUT_PATH)  # y from -40 to 40, over the first box only
        extent = logic.computeAutoExtent(model)

        options = CutOptions()
        fragments = logic.cutPolyData(model, [logic.buildSheetPolyData(path, np.array(self.DOWN), extent)], options)
        self.assertEqual(len(fragments), 4)  # both boxes split

        options.endExtension = 15.0  # the cut stops at y = +/-55, in the gap between the boxes
        sheet = logic.buildSheetPolyData(path, np.array(self.DOWN), extent, endExtension=15.0)
        self.assertFalse(logic.sheetEndsInModel(model, sheet, logic.effectiveKerfWidth(options)))
        fragments = logic.cutPolyData(model, [sheet], options)
        self.assertEqual(len(fragments), 3)  # the second box is whole
        for fragment in fragments:
            self._assertWatertight(fragment)
        self.assertTrue(any(abs(self._volume(fragment) - 40.0 ** 3) < 1.0 for fragment in fragments))

    def test_endInsideBone_refused(self):
        """A cut that would end inside the bone beyond its line ends (a slot) is refused with a
        clear message, rather than leaving bone segments that are not closed."""
        logic = OsteotomyCutsLogic()
        box = self._box()
        path = np.array([[1.3, -20.0, 50.0], [1.3, 20.0, 50.0]])
        sheet = logic.buildSheetPolyData(path, np.array(self.DOWN), logic.computeAutoExtent(box), endExtension=5.0)
        options = CutOptions()
        options.endExtension = 5.0
        self.assertTrue(logic.sheetEndsInModel(box, sheet, logic.effectiveKerfWidth(options)))
        with self.assertRaises(ValueError) as raised:
            logic.cutPolyData(box, [sheet], options)
        self.assertIn("Past line ends", str(raised.exception))
        # Ending 0.2 mm outside the bone (coarse mesh: no vertex near) is clear of a 0.1 mm kerf,
        # but not of a 1 mm kerf, whose rounded end reaches the bone
        near = logic.buildSheetPolyData(np.array([[1.3, -49.8, 50.0], [1.3, 49.8, 50.0]]), np.array(self.DOWN),
                                        logic.computeAutoExtent(box), endExtension=0.4)
        self.assertFalse(logic.sheetEndsInModel(box, near, 0.1))
        self.assertTrue(logic.sheetEndsInModel(box, near, 1.0))

    def test_cutOutline(self):
        """The outline follows the cut where it meets the surface, and stops where the sheet stops."""
        logic = OsteotomyCutsLogic()
        box = self._box()
        extent = logic.computeAutoExtent(box)
        path, down = np.array(self.X_CUT_PATH), np.array(self.DOWN)

        # Through-cut: round the box, on its top, front, bottom and back faces
        through = logic.computeCutOutline(box, logic.buildSheetPolyData(path, down, extent))
        self.assertAlmostEqual(self._lineLength(through), 400.0, delta=1e-6)
        np.testing.assert_allclose(self._points(through)[:, 0], 1.3, atol=1e-6)

        # 10 mm deep: across the top face and 10 mm down the front and back faces
        groove = logic.computeCutOutline(box, logic.buildSheetPolyData(path, down, extent, depth=10.0))
        self.assertAlmostEqual(self._lineLength(groove), 120.0, delta=1e-6)
        self.assertAlmostEqual(groove.GetBounds()[4], 40.0, delta=1e-6)

        # Stopping 5 mm past the line ends (y = +/-40): on the top and bottom faces only, |y| <= 45
        slot = logic.computeCutOutline(box, logic.buildSheetPolyData(path, down, extent, endExtension=5.0))
        self.assertAlmostEqual(np.abs(self._points(slot)[:, 1]).max(), 45.0, delta=1e-6)
        self.assertAlmostEqual(self._lineLength(slot), 180.0, delta=1e-6)

        # A sheet that misses the model gives no lines
        missing = logic.computeCutOutline(box, logic.buildSheetPolyData(path + [200.0, 0.0, 0.0], down, 10.0))
        self.assertEqual(missing.GetNumberOfLines(), 0)

    def test_outlinePreview(self):
        """The preview shows the cut outline on the model, hidden with the sheet."""
        logic = OsteotomyCutsLogic()
        model = self._addModel(self._box(), "Box")
        curve = self._addCurve(self.X_CUT_PATH, "CutA")
        parameterNode = self._configure(model, curve)
        parameterNode.options.depth = 10.0

        self.assertIsNotNone(logic.updateSheetModel(parameterNode))
        outlineNode = parameterNode.outlineModel
        self.assertIsNotNone(outlineNode)
        self.assertTrue(outlineNode.GetHideFromEditors())
        self.assertFalse(outlineNode.GetSelectable())
        self.assertTrue(outlineNode.GetDisplayNode().GetVisibility())
        self.assertAlmostEqual(self._lineLength(outlineNode.GetPolyData()), 120.0, delta=1e-6)

        # The outline follows a moved model (world coordinates)
        transform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
        matrix = vtk.vtkMatrix4x4()
        matrix.SetElement(2, 3, -5.0)  # top face at z = 45: the 10 mm deep sheet enters 5 mm
        transform.SetMatrixTransformToParent(matrix)
        model.SetAndObserveTransformNodeID(transform.GetID())
        logic.updateSheetModel(parameterNode)
        self.assertAlmostEqual(self._lineLength(outlineNode.GetPolyData()), 110.0, delta=1e-6)

        parameterNode.livePreview = False
        self.assertIsNone(logic.updateSheetModel(parameterNode))
        self.assertFalse(outlineNode.GetDisplayNode().GetVisibility())
        self.assertEqual(len(slicer.util.getNodes("CutOutline*")), 1)

    #
    # Release Part 1: solid bone models
    #

    @staticmethod
    def _holedSphere(radius: float, holeRadius: float) -> vtk.vtkPolyData:
        """Sphere at the origin with a round hole of about holeRadius at +x."""
        hole = vtk.vtkSphere()
        hole.SetCenter(radius, 0.0, 0.0)
        hole.SetRadius(holeRadius)
        clip = vtk.vtkClipPolyData()
        clip.SetInputData(OsteotomyCutsTest._sphere(radius))
        clip.SetClipFunction(hole)
        clip.Update()
        return clip.GetOutput()

    def _hollowSphere(self) -> vtk.vtkPolyData:
        """Shell between spheres of radius 30 and 20, its cavity open to the outside through a
        tunnel 1 mm wide along +x (an open, non-solid bone model)."""
        tube = vtk.vtkCylinderSource()  # along y
        tube.SetRadius(0.5)
        tube.SetHeight(10.0)
        tube.SetResolution(16)
        tube.CappingOff()
        transform = vtk.vtkTransform()
        transform.Translate(25.0, 0.0, 0.0)
        transform.RotateZ(-90.0)
        moved = vtk.vtkTransformPolyDataFilter()
        moved.SetInputConnection(tube.GetOutputPort())
        moved.SetTransform(transform)
        moved.Update()
        return self._append(self._holedSphere(30.0, 0.5), self._holedSphere(20.0, 0.5), moved.GetOutput())

    def _boxWithMarrow(self) -> vtk.vtkPolyData:
        """40 mm box with an internal sphere surface of radius 10 (like the inner cortex)."""
        return self._append(self._box(20.0, 10), self._sphere(10.0, resolution=32))

    def _assertClosedSolid(self, polyData: vtk.vtkPolyData, expectedVolume: float) -> None:
        logic = OsteotomyCutsLogic()
        merged = logic.mergeCoincidentPoints(polyData)
        self.assertEqual(self._openEdgeCount(merged), 0)
        self.assertEqual(logic._connectedRegions(merged)[3], 1)
        self.assertAlmostEqual(self._volume(merged), expectedVolume, delta=0.03 * expectedVolume)

    def test_solid_hollowSphere(self):
        """A hollow, open sphere becomes one closed solid of the outer sphere's volume; the
        input is not modified."""
        logic = OsteotomyCutsLogic()
        model = self._hollowSphere()
        self.assertGreater(self._openEdgeCount(model), 0)
        pointsBefore = self._points(model).copy()
        solid = logic.makeSolidPolyData(model, voxelSize=0.5)
        np.testing.assert_array_equal(self._points(model), pointsBefore)
        self._assertClosedSolid(solid, 4.0 / 3.0 * np.pi * 30.0 ** 3)
        self.assertIsNotNone(solid.GetPointData().GetNormals())

    def test_solid_sealsHole(self):
        """A 1 mm hole is sealed; the surface stays within about a voxel of the original."""
        logic = OsteotomyCutsLogic()
        solid = logic.makeSolidPolyData(self._holedSphere(20.0, 0.5), voxelSize=0.5)
        self._assertClosedSolid(solid, 4.0 / 3.0 * np.pi * 20.0 ** 3)
        radii = self._radii(solid)
        self.assertLess(np.max(np.abs(radii - 20.0)), 0.5)

    def test_solid_errors(self):
        """Too many voxels (memory guard) and invalid settings raise ValueError before any work."""
        logic = OsteotomyCutsLogic()
        with self.assertRaises(ValueError) as context:
            logic.makeSolidPolyData(self._box(), voxelSize=0.02)
        self.assertIn("detail", str(context.exception))
        for voxelSize, closing in ((0.0, 1.5), (0.5, -1.0)):
            with self.assertRaises(ValueError):
                logic.makeSolidPolyData(self._box(), voxelSize=voxelSize, closingMm=closing)
        with self.assertRaises(ValueError):
            logic.makeSolidPolyData(vtk.vtkPolyData())

    def test_solid_cutOption(self):
        """With "treat bone as solid", bone segments are closed and filled, marked solid, and
        cutting them again uses them as they are; without it the internal surface is kept."""
        logic = OsteotomyCutsLogic()
        model = self._addModel(self._boxWithMarrow(), "Box")
        curve = self._addCurve([[1.3, -30.0, 20.0], [1.3, 30.0, 20.0]], "CutA")
        parameterNode = self._configure(model, curve)

        nodes = logic.applyCut(parameterNode)  # option off: as before
        self.assertEqual(len(nodes), 2)
        for node in nodes:
            merged = logic.mergeCoincidentPoints(node.GetPolyData())
            self.assertEqual(logic._connectedRegions(merged)[3], 2)  # box half + marrow half
            self.assertFalse(logic.isSolidModel(node))
        self.assertEqual(logic._solidCache, {})

        parameterNode.treatBoneAsSolid = True
        parameterNode.solidVoxelSize = 0.5
        nodes = logic.applyCut(parameterNode)
        self.assertEqual(len(nodes), 2)
        volumes = sorted(self._volume(node.GetPolyData()) for node in nodes)
        for node, expected in zip(sorted(nodes, key=lambda n: self._volume(n.GetPolyData())),
                                  (18.7 * 1600.0, 21.3 * 1600.0)):
            self._assertClosedSolid(node.GetPolyData(), expected)
            self.assertTrue(logic.isSolidModel(node))
        self.assertLess(volumes[0], volumes[1])
        self.assertFalse(logic.isSolidModel(model))
        solid = logic.getSolidPolyData(model, 0.5, 1.5)
        logic.applyCut(parameterNode)  # re-applied: the cached solid is reused
        self.assertIs(logic.getSolidPolyData(model, 0.5, 1.5), solid)

        # A solid bone segment is cut as it is (no new solid model)
        right = next(node for node in logic.getCurveResult(curve) if node.GetPolyData().GetBounds()[0] > 0.0)
        curveB = self._addCurve([[5.0, 2.7, 20.0], [15.0, 2.7, 20.0]], "CutB")
        parameterNode.inputModel = right
        parameterNode.cutCurve = curveB
        self.assertEqual(len(logic.applyCut(parameterNode)), 2)
        self.assertEqual(list(logic._solidCache), [model.GetID()])

    def test_createSolidModel(self):
        """"Create solid bone model" adds a closed, marked copy and hides the original, unchanged."""
        logic = OsteotomyCutsLogic()
        model = self._addModel(self._boxWithMarrow(), "Box")
        pointsBefore = self._points(model.GetPolyData()).copy()
        solidNode = logic.createSolidModel(model, 0.5, 1.5)

        self.assertEqual(solidNode.GetName(), "Box_solid")
        self.assertTrue(logic.isSolidModel(solidNode))
        self.assertEqual(solidNode.GetAttribute(SOLID_ATTRIBUTE), "1")
        self._assertClosedSolid(solidNode.GetPolyData(), 40.0 ** 3)
        self.assertFalse(self._isVisible(model))
        self.assertTrue(self._isVisible(solidNode))
        np.testing.assert_array_equal(self._points(model.GetPolyData()), pointsBefore)
        # The node's mesh is its own: changing it leaves the cached solid alone
        self.assertIsNot(solidNode.GetPolyData(), logic.getSolidPolyData(model, 0.5, 1.5))

    #
    # Release Part 0d: osteotomy of several lines
    #

    # "Le Fort I" on the test box: a horizontal groove cut from the front (+y), 60 mm deep, and a
    # vertical line on the right face extruded to the left (-x) through the box, 10 mm in front of
    # the groove's end
    LE_FORT_LINES = ([[-30.0, -50.0, 10.3], [30.0, -50.0, 10.3]], [[50.0, 5.3, 10.3], [50.0, 5.3, -40.0]])
    LE_FORT_DIRECTIONS = ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))

    def _leFortCut(self, kerfWidth: float, grooveDepth: float = 60.0) -> list:
        logic = OsteotomyCutsLogic()
        box = self._box()
        extent = logic.computeAutoExtent(box)
        first, second = (np.array(line) for line in self.LE_FORT_LINES)
        horizontal = logic.buildSheetPolyData(first, np.array(self.LE_FORT_DIRECTIONS[0]), extent, depth=grooveDepth)
        vertical = logic.buildSheetPolyData(second, np.array(self.LE_FORT_DIRECTIONS[1]), extent)
        options = CutOptions()
        options.kerfWidth = kerfWidth
        options.depth = grooveDepth
        verticalOptions = CutOptions()
        verticalOptions.kerfWidth = kerfWidth
        sides = logic.stopSides(second, [horizontal])
        return logic.cutPolyData(box, [horizontal, vertical], [options, verticalOptions],
                                 stops=[[], [(horizontal, sides[0])]])

    def test_osteotomy_stopsAtEarlierLine(self):
        """The vertical line stops at the horizontal groove: the front-lower block comes off,
        closed, and the bone above the groove is not cut (with a blade and the ideal blade)."""
        for kerfWidth in (1.0, 0.0):
            fragments = self._leFortCut(kerfWidth)
            self.assertEqual(len(fragments), 2, msg=f"kerf {kerfWidth}")
            for fragment in fragments:
                self._assertWatertight(fragment)
            big, block = sorted(fragments, key=self._volume, reverse=True)
            half = max(kerfWidth, OsteotomyCutsLogic.LIMITED_IDEAL_KERF_WIDTH) / 2.0
            np.testing.assert_allclose(block.GetBounds(), (-50.0, 50.0, -50.0, 5.3 - half, -50.0, 10.3 - half),
                                       atol=1e-3)
            self.assertAlmostEqual(self._volume(block), 100.0 * (55.3 - half) * (60.3 - half), delta=1.0)
            selector = vtk.vtkSelectEnclosedPoints()
            selector.Initialize(big)
            self.assertTrue(selector.IsInsideSurface(0.0, 5.3, 30.0))  # above the groove: not cut
            self.assertFalse(selector.IsInsideSurface(0.0, 5.3, -30.0))  # in the vertical cut
            selector.Complete()

    def test_osteotomy_junctionBeyondEarlierCut(self):
        """A later line meeting bone across an earlier line beyond that line's cut is refused."""
        with self.assertRaises(ValueError) as context:
            self._leFortCut(1.0, grooveDepth=30.0)
        self.assertIn("far side of an earlier line", str(context.exception))

    def test_osteotomy_lines(self):
        """Each line keeps its own settings; lines grouped into an osteotomy are previewed,
        applied (from any of its lines) and undone together; all of it survives a scene reload."""
        import os
        logic = OsteotomyCutsLogic()
        model = self._addModel(self._box(), "Box")
        horizontal = self._addCurve(self.LE_FORT_LINES[0], "LeFort")
        vertical = self._addCurve(self.LE_FORT_LINES[1], "PtMax")

        # Settings per line: the vertical line first, then a new line takes them and changes them
        parameterNode = self._configure(model, vertical, direction=self.LE_FORT_DIRECTIONS[1])
        parameterNode.options.kerfWidth = 1.0
        logic.storeLineSettings(parameterNode)
        parameterNode.options.depth = 20.0
        parameterNode.options.endExtension = 5.0
        parameterNode.cutCurve = horizontal
        # A new line keeps the saw settings, but not the other line's depth, reach or direction
        self.assertFalse(logic.loadLineSettings(parameterNode))
        self.assertEqual(parameterNode.options.kerfWidth, 1.0)
        self.assertEqual(parameterNode.options.depth, 0.0)
        self.assertEqual(parameterNode.options.endExtension, 0.0)
        self.assertFalse(isDirectionCaptured(parameterNode.viewDirection))
        parameterNode.viewDirection = self.LE_FORT_DIRECTIONS[0]
        parameterNode.options.depth = 60.0
        logic.storeLineSettings(parameterNode)
        parameterNode.cutCurve = vertical
        self.assertTrue(logic.loadLineSettings(parameterNode))
        self.assertEqual(parameterNode.options.depth, 0.0)
        np.testing.assert_allclose(parameterNode.viewDirection, self.LE_FORT_DIRECTIONS[1])
        self.assertEqual(parameterNode.options.kerfWidth, 1.0)

        # Group: the vertical line joins the horizontal line's osteotomy
        self.assertEqual(self._ids(logic.getOsteotomyLines(vertical)), [vertical.GetID()])
        logic.setGroupLines(horizontal, [vertical])
        self.assertEqual(self._ids(logic.getOsteotomyLines(vertical)), [horizontal.GetID(), vertical.GetID()])
        self.assertEqual(logic.findFirstLine(vertical).GetID(), horizontal.GetID())
        self.assertIsNone(logic.validateInputs(parameterNode))

        # Preview: both sheets; the vertical line's outline stops at the horizontal cut (no line
        # on the top face). Horizontal: front 100 + sides 2 x 60; vertical: sides 2 x 60.3 + bottom 100
        logic.updateSheetModel(parameterNode)
        self.assertEqual(parameterNode.sheetModel.GetPolyData().GetNumberOfCells(),
                         sum(line.sheet.GetNumberOfCells() for line in logic.buildOsteotomy(parameterNode)))
        outline = parameterNode.outlineModel.GetPolyData()
        self.assertAlmostEqual(self._lineLength(outline), 220.0 + 220.6, delta=5.0)
        self.assertLess(outline.GetBounds()[5], 12.0)

        # Applied from the vertical line: one result, belonging to the first line
        nodes = logic.applyCut(parameterNode)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(self._ids(logic.getCurveResult(horizontal)), self._ids(nodes))
        self.assertEqual(logic.getCurveResult(vertical), [])
        self.assertTrue(all(node.GetName().startswith("Box_LeFort_") for node in nodes))
        for node in nodes:
            self._assertWatertight(node.GetPolyData())

        # Scene round trip keeps the settings and the group
        scenePath = os.path.join(slicer.app.temporaryPath, "OsteotomyCutsLines.mrb")
        try:
            self.assertTrue(slicer.util.saveScene(scenePath))
            slicer.mrmlScene.Clear()
            slicer.util.loadScene(scenePath)
        finally:
            if os.path.exists(scenePath):
                os.remove(scenePath)
        horizontal = slicer.util.getNode("LeFort")
        vertical = slicer.util.getNode("PtMax")
        self.assertEqual(self._ids(logic.getGroupLines(horizontal)), [vertical.GetID()])
        self.assertEqual(logic.getLineSettings(horizontal).options.depth, 60.0)
        np.testing.assert_allclose(logic.getLineSettings(vertical).viewDirection, self.LE_FORT_DIRECTIONS[1])
        self.assertEqual(len(logic.getCurveResult(horizontal)), 2)

        # Undo through the first line removes everything
        logic.removeCutResult(logic.findFirstLine(vertical))
        self.assertEqual(logic.getCurveResult(horizontal), [])
        self.assertTrue(self._isVisible(slicer.util.getNode("Box")))

        # A line is in one osteotomy only
        other = self._addCurve([[0.0, -40.0, 50.0], [0.0, 40.0, 50.0]], "Other")
        logic.setGroupLines(other, [vertical])
        self.assertEqual(logic.getGroupLines(horizontal), [])
        self.assertEqual(logic.findFirstLine(vertical).GetID(), other.GetID())

    def test_joinSmallSegments(self):
        """A small loose piece is joined to the large segment it touches; 0 keeps every piece."""
        logic = OsteotomyCutsLogic()
        left, right = self._sphere(20.0), self._sphere(20.0, centre=(50.0, 0.0, 0.0))
        piece = self._sphere(2.0, centre=(22.5, 0.0, 0.0), resolution=16)  # 0.5 mm from the left sphere
        fragments = [left, right, piece]

        kept, joined = logic.joinSmallSegments(fragments, 0.0)
        self.assertEqual((len(kept), joined), (3, 0))
        kept, joined = logic.joinSmallSegments(fragments, 0.02, contactDistance=1.0)
        self.assertEqual((len(kept), joined), (2, 1))
        withPiece = next(mesh for mesh in kept if mesh.GetBounds()[1] > 20.0 and mesh.GetBounds()[0] < 0.0)
        self.assertEqual(withPiece.GetNumberOfPoints(), left.GetNumberOfPoints() + piece.GetNumberOfPoints())
        self.assertEqual(logic._connectedRegions(withPiece)[3], 2)

    #
    # Release Part 2: model quality
    #

    def test_assessModel(self):
        """Closed, holed, crossing, several pieces and internal surfaces are told apart."""
        logic = OsteotomyCutsLogic()
        box = logic.assessModel(self._box())
        self.assertTrue(box.isClosed)
        self.assertEqual((box.pieceCount, box.enclosedShellCount), (1, 0))

        holed = logic.assessModel(self._holedSphere(20.0, 2.0))
        self.assertFalse(holed.isClosed)
        self.assertGreater(holed.openEdges, 0)

        marrow = logic.assessModel(self._boxWithMarrow())
        self.assertTrue(marrow.isClosed)
        self.assertEqual((marrow.pieceCount, marrow.enclosedShellCount), (2, 1))

        apart = logic.assessModel(self._append(self._sphere(10.0), self._sphere(10.0, centre=(30.0, 0.0, 0.0))))
        self.assertEqual((apart.pieceCount, apart.enclosedShellCount), (2, 0))

        # Three triangles on one edge: surfaces crossing
        fan = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        for point in ((0, 0, 0), (0, 0, 1), (1, 0, 0), (0, 1, 0), (-1, 0, 0)):
            points.InsertNextPoint(point)
        fan.SetPoints(points)
        cells = vtk.vtkCellArray()
        for third in (2, 3, 4):
            cells.InsertNextCell(3, [0, 1, third])
        fan.SetPolys(cells)
        self.assertGreater(logic.assessModel(fan).nonManifoldEdges, 0)

    def test_assessSegments(self):
        """Bone segments are marked closed or not, with their volume."""
        logic = OsteotomyCutsLogic()
        model = self._addModel(self._box(), "Box")
        parameterNode = self._configure(model, self._addCurve(self.X_CUT_PATH, "CutA"))
        nodes = logic.applyCut(parameterNode)
        self.assertEqual([result.isClosed for result in logic.lastSegmentQuality], [True, True])
        self.assertAlmostEqual(sum(result.volume for result in logic.lastSegmentQuality), 1e6, delta=10.0)
        for node, result in zip(nodes, logic.lastSegmentQuality):
            self.assertEqual(node.GetAttribute("OsteotomyCuts.Watertight"), "true")
            self.assertAlmostEqual(float(node.GetAttribute("OsteotomyCuts.Volume_mm3")), result.volume, delta=0.1)
            self.assertEqual(result.name, node.GetName())

        parameterNode.options.capCutFaces = False
        nodes = logic.applyCut(parameterNode)
        self.assertEqual([result.isClosed for result in logic.lastSegmentQuality], [False, False])
        self.assertEqual({node.GetAttribute("OsteotomyCuts.Watertight") for node in nodes}, {"false"})

        # A model made solid for the cut is not checked (it is closed); a hollow one is
        parameterNode.treatBoneAsSolid = True
        self.assertIsNone(logic.assessModelToCut(parameterNode))
        parameterNode.treatBoneAsSolid = False
        self.assertTrue(logic.assessModelToCut(parameterNode).isClosed)

    #
    # Release Part 3: structures to protect
    #

    @staticmethod
    def _canal(x: float, z: float = 0.0, radius: float = 1.5) -> vtk.vtkPolyData:
        """Closed cylinder along y (a canal model) through (x, 0, z)."""
        cylinder = vtk.vtkCylinderSource()  # along y
        cylinder.SetRadius(radius)
        cylinder.SetHeight(80.0)
        cylinder.SetResolution(96)
        cylinder.SetCenter(x, 0.0, z)
        cylinder.CappingOn()
        triangles = vtk.vtkTriangleFilter()
        triangles.SetInputConnection(cylinder.GetOutputPort())
        triangles.Update()
        return triangles.GetOutput()

    def _planarLines(self, kerfWidth: float = 0.0, depth: float = 0.0) -> list:
        """One osteotomy line: the plane x = 1.3 through the test box, cut from the top."""
        logic = OsteotomyCutsLogic()
        box = self._box()
        options = CutOptions()
        options.kerfWidth = kerfWidth
        options.depth = depth
        sheet = logic.buildSheetPolyData(np.array(self.X_CUT_PATH), np.array(self.DOWN),
                                         logic.computeAutoExtent(box), depth=depth if depth > 0 else None)
        curve = self._addCurve(self.X_CUT_PATH, "Line")
        return [OsteotomyLine(curve, sheet, options, [])]

    def _structure(self, polyData_or_points, name: str, safeDistance: float = 2.0,
                   radius: float = 1.5) -> ProtectedStructure:
        logic = OsteotomyCutsLogic()
        if isinstance(polyData_or_points, vtk.vtkPolyData):
            node = self._addModel(polyData_or_points, name)
        else:
            node = self._addCurve(polyData_or_points, name)
        structure = logic.addProtectedStructure(node)
        structure.safeDistance, structure.radius = safeDistance, radius
        logic.setProtectedStructure(structure)
        return structure

    def test_clearance_canal(self):
        """A canal 5 mm from a planar cut (surface 3.5 mm away): the clearance is exact to 0.1 mm,
        with the ideal blade and with a 1 mm blade; a centreline curve with a radius gives the same."""
        logic = OsteotomyCutsLogic()
        box = self._box()
        canal = self._structure(self._canal(6.3), "Mandibular canal")
        self.assertEqual(canal.category, StructureCategory.NERVE)
        nerve = self._structure([[6.3, -40.0, 0.0], [6.3, 40.0, 0.0]], "IAN")
        for kerfWidth, expected in ((0.0, 3.5), (1.0, 3.0)):
            results = logic.checkClearances(self._planarLines(kerfWidth), box, [canal, nerve])
            for result in results:
                self.assertAlmostEqual(result.clearance, expected, delta=0.1,
                                       msg=f"{result.structure.node.GetName()}, kerf {kerfWidth}")
                self.assertEqual(result.status, ClearanceStatus.SAFE)
                self.assertAlmostEqual(result.closestPoint[0], 1.3, delta=0.01)

    def test_clearance_statuses(self):
        """Too close, cut entering the structure, a cut stopping short of it, and parts of the
        cut in air ignored; each structure with its own safe distance."""
        logic = OsteotomyCutsLogic()
        box = self._box()
        near = self._structure(self._canal(6.3), "Near canal", safeDistance=4.0)
        tooth = self._structure(self._sphere(2.0, centre=(-10.0, 0.0, 10.0), resolution=32), "Tooth root",
                                safeDistance=1.0)
        self.assertEqual(tooth.category, StructureCategory.TOOTH)
        through = self._structure(self._canal(1.3, z=-20.0), "Canal in the cut", safeDistance=2.0)
        inAir = self._structure(self._sphere(2.0, centre=(1.3, 0.0, 70.0), resolution=32), "Above the bone")
        results = {r.structure.node.GetName(): r for r in logic.checkClearances(
            self._planarLines(), box, [near, tooth, through, inAir])}
        self.assertEqual(results["Near canal"].status, ClearanceStatus.TOO_CLOSE)
        self.assertEqual(results["Tooth root"].status, ClearanceStatus.SAFE)
        self.assertAlmostEqual(results["Tooth root"].clearance, 11.3 - 2.0, delta=0.1)
        self.assertEqual(results["Canal in the cut"].status, ClearanceStatus.ENTERS)
        self.assertAlmostEqual(results["Canal in the cut"].clearance, -1.5, delta=0.1)
        self.assertEqual(results["Above the bone"].status, ClearanceStatus.SAFE)
        self.assertAlmostEqual(results["Above the bone"].clearance, 70.0 - 50.0 - 2.0, delta=0.15)

        # A cut 30 mm deep from the top (z = 50) ends at z = 20, 40 mm above the canal's axis
        # (the ideal blade of a limited cut removes 0.1 mm: half is 0.05 mm)
        stopped = logic.checkClearances(self._planarLines(depth=30.0), box, [through])[0]
        self.assertEqual(stopped.status, ClearanceStatus.SAFE)
        self.assertAlmostEqual(stopped.clearance, 40.0 - 1.5 - 0.05, delta=0.1)

        # Disabled structures are not checked
        through.enabled = False
        self.assertEqual(logic.checkClearances(self._planarLines(), box, [through]), [])

    def test_clearance_mrml(self):
        """Structures are found in the scene with their settings, which survive a scene reload;
        the check uses the osteotomy and records the results on the bone segments."""
        import os
        logic = OsteotomyCutsLogic()
        model = self._addModel(self._box(), "Box")
        parameterNode = self._configure(model, self._addCurve(self.X_CUT_PATH, "CutA"))
        teeth = logic.addProtectedStructure(self._addModel(self._canal(6.3), "Lower Teeth"))
        self.assertEqual((teeth.category, teeth.safeDistance), (StructureCategory.TOOTH, 1.0))
        with self.assertRaises(ValueError):
            logic.addProtectedStructure(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode"))

        results = logic.checkClearancesForParameters(parameterNode)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].clearance, 3.5, delta=0.1)
        logic.updateClearanceDisplay(parameterNode, results)
        self.assertEqual(parameterNode.clearanceMarkups.GetNumberOfControlPoints(), 1)
        self.assertIn("3.5 mm", parameterNode.clearanceMarkups.GetNthControlPointLabel(0))

        nodes = logic.applyCut(parameterNode)
        logic.recordSafetyResults(nodes, results, override=True)
        stored = json.loads(nodes[0].GetAttribute("OsteotomyCuts.SafetyResults"))
        self.assertEqual((stored[0]["structure"], stored[0]["status"]), ("Lower Teeth", "safe"))
        self.assertEqual(nodes[0].GetAttribute("OsteotomyCuts.SafetyOverride"), "true")

        teeth.safeDistance, teeth.enabled = 0.5, False
        logic.setProtectedStructure(teeth)
        scenePath = os.path.join(slicer.app.temporaryPath, "OsteotomyCutsStructures.mrb")
        try:
            self.assertTrue(slicer.util.saveScene(scenePath))
            slicer.mrmlScene.Clear()
            slicer.util.loadScene(scenePath)
        finally:
            if os.path.exists(scenePath):
                os.remove(scenePath)
        structures = logic.getProtectedStructures()
        self.assertEqual([(s.node.GetName(), s.category, s.safeDistance, s.enabled) for s in structures],
                         [("Lower Teeth", StructureCategory.TOOTH, 0.5, False)])
        logic.removeProtectedStructure(structures[0].node)
        self.assertEqual(logic.getProtectedStructures(), [])
