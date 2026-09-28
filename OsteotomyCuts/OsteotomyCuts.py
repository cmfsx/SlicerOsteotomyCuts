import enum
import logging
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
            # exit() may have removed the observer already.
            if self.hasObserver(self._parameterNode.parameterNode, vtk.vtkCommand.ModifiedEvent,
                                self._updateGuiFromParameterNode):
                self.removeObserver(self._parameterNode.parameterNode, vtk.vtkCommand.ModifiedEvent,
                                    self._updateGuiFromParameterNode)
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
    # Fragment normals are not smoothed across edges sharper than this (degrees), so that cut
    # faces are shaded flat
    NORMALS_FEATURE_ANGLE = 30.0

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
                   arrayName: str = "SheetDistance") -> tuple[vtk.vtkPolyData, bool]:
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
        :return: (remaining triangle mesh, whether any material was removed).
        :raises ValueError: for a non-positive kerf width.
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
        if np.all(keep):
            return mesh, False

        points = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData()).astype(float)
        triangles = numpy_support.vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3).astype(np.int64)
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
        # loops cannot tell such points apart
        lengths = np.linalg.norm(points[cutEdges[:, 0]] - points[cutEdges[:, 1]], axis=1)
        diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        short = lengths < self.RIM_COLLAPSE_FRACTION * diagonal
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
                    options: CutOptions,
                    progressCallback: Optional[ProgressCallback] = None) -> list[vtk.vtkPolyData]:
        """Cut a mesh with one or more sheets into separate fragments (headless core).

        Each sheet is applied in turn to every current piece, then the pieces are split into
        connected fragments and enclosed pieces are merged into their host. Several sheets let
        Phase 3 cut several osteotomies at once. The input mesh is not modified.

        With kerfWidth = 0 each piece is split at signed distance 0 into its two sides. With a
        kerf, material within kerfWidth / 2 of the sheet is removed (removeKerf) and the sides
        are found afterwards from the signed distance kept per sheet.

        With capCutFaces, the cut faces a sheet leaves are capped (capCutFaces) right
        after that sheet, so later sheets cut closed meshes and the fragments are watertight.
        A groove left by a depth-limited cut is lined the same way, walls and rounded floor.
        Fragments then get display normals split at sharp edges (computeDisplayNormals).

        :param polyData: model mesh in world coordinates.
        :param sheets: cutting sheets from buildSheetPolyData.
        :param options: cut options.
        :param progressCallback: called with (percent, message) between stages.
        :return: fragment meshes, largest first. A depth-limited kerf cut may leave one
            fragment (a groove).
        :raises ValueError: if there is no sheet, the mesh has no polygons, or a kerf sheet
            does not reach the model.
        """
        report = progressCallback or (lambda percent, message: None)
        if not sheets:
            raise ValueError(_("At least one cutting sheet is needed."))
        # A fragment of an earlier cut has duplicate points along sharp edges (display normals)
        triangles = self.mergeCoincidentPoints(self._ensureTriangles(polyData))
        if triangles.GetNumberOfCells() == 0:
            raise ValueError(_("The model has no surface polygons to cut."))
        cap = options.capCutFaces
        halfKerf = options.kerfWidth / 2.0
        uncapped = 0

        report(5, _("Finding internal shells..."))
        labelled = self.labelEnclosedComponents(triangles, options.minFragmentFraction)
        sides = [(labelled, ())]
        for sheetIndex, sheet in enumerate(sheets):
            report(15 + int(60 * sheetIndex / len(sheets)), _("Cutting..."))
            nextSides = []
            removedAny = False
            sheetMap = SheetParameterisation(sheet, halfKerf) if cap else None
            for mesh, signature in sides:
                withDistance = self.computeSheetDistance(mesh, sheet)
                maxEdgeLength = self.effectiveRefineEdgeLength(options)
                if maxEdgeLength > 0:
                    withDistance = self.refineNearSheet(withDistance, sheet, maxEdgeLength,
                                                        options.kerfWidth / 2.0 + maxEdgeLength)
                if options.kerfWidth > 0:
                    sideArray = vtk.vtkDoubleArray()
                    sideArray.DeepCopy(withDistance.GetPointData().GetArray("SheetDistance"))
                    sideArray.SetName(f"{SHEET_SIDE_PREFIX}{sheetIndex}")
                    withDistance.GetPointData().AddArray(sideArray)
                    remaining, removed = self.removeKerf(withDistance, sheet, options.kerfWidth)
                    removedAny = removedAny or removed
                    if cap and removed:
                        remaining, failed = self.capCutFaces(remaining, sheetMap)
                        uncapped += failed
                    if remaining.GetNumberOfCells() > 0:
                        nextSides.append((remaining, signature))
                    continue
                positive, negative = self.splitByDistance(withDistance, options)
                for part, side in ((positive, 1), (negative, -1)):
                    if part.GetNumberOfCells() > 0:
                        if cap:
                            part, failed = self.capCutFaces(part, sheetMap)
                            uncapped += failed
                        nextSides.append((part, signature + (side,)))
            if options.kerfWidth > 0 and not removedAny:
                raise ValueError(_("Cutting sheet {index} does not reach the model. Check the cut path, "
                                   "direction and depth.").format(index=sheetIndex + 1))
            sides = nextSides

        report(75, _("Separating fragments..."))
        pieces = []
        for mesh, signature in sides:
            pieces.extend(self.extractFragments(mesh, signature))
        fragments = self.mergeEnclosedPieces(pieces, options.minFragmentFraction * labelled.GetNumberOfPoints())

        for fragment in fragments:
            sideArrays = [f"{SHEET_SIDE_PREFIX}{i}" for i in self._sheetSideIndices(fragment)]
            for arrayName in ["ComponentId", "HostComponentId", "SheetDistance"] + sideArrays:
                fragment.GetPointData().RemoveArray(arrayName)
        if uncapped:
            logging.warning(f"OsteotomyCuts: {uncapped} cut face(s) could not be capped; "
                            "the fragments are not watertight there.")
        if cap:
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
        :return: the new fragment model nodes, largest first (a single node for a groove, a
            depth-limited cut that does not separate the model).
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
        # A depth-limited cut may only cut a groove (cutPolyData has checked that it removed bone)
        isGroove = parameterNode.options.depth > 0 and parameterNode.options.kerfWidth > 0
        if len(fragments) < (1 if isGroove else 2):
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
        """Depth needs a kerf; the preview sheet follows the depth; a groove gives one fragment."""
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
