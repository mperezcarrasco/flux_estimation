"""AER_AI plume-boundary definition and masking utilities.

Supports the plume-boundary-inspection notebook: given a fixed AI
threshold, AI>threshold pixels are grouped into spatially-coherent blobs
(DBSCAN on lat/lon), the physically real plume is identified as one or more
of those blobs, and small isolated blobs ("noise plumes") are dropped by
exclusion. The kept blobs' hulls are persisted as GeoJSON so a boundary
defined once for a given date can be reloaded by other notebooks.

Reference notebook:
    notebooks/plume_boundary_inspection.ipynb
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import shapely
from shapely.geometry import MultiPoint, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN

EARTH_RADIUS_KM: float = 6_371.0  # WGS-84 mean


@dataclass(frozen=True)
class ThresholdResult:
    """Robust background-based AI threshold: median + k * MAD.

    Attributes
    ----------
    threshold : AI value above which pixels are considered plume candidates.
    median    : domain-wide median AI (robust background estimate).
    mad       : domain-wide median absolute deviation of AI.
    k         : MAD multiplier used to compute `threshold` (or the
        equivalent MAD-multiple if `threshold` was set manually).
    """

    threshold: float
    median: float
    mad: float
    k: float


def robust_ai_threshold(ai: np.ndarray, k: float) -> ThresholdResult:
    """Background-based AI threshold via median + k * MAD.

    Uses robust statistics over the *entire* input array rather than a
    hand-picked background region: plume pixels are a small minority of
    any domain-wide AER_AI sample, so the median and MAD are dominated by
    background pixels and largely insensitive to the plume tail.

    Parameters
    ----------
    ai : (n,) AER_AI values. Must be finite (filter NaNs before calling).
    k  : MAD multiplier (> 0). Larger k -> higher, more conservative threshold.

    Returns
    -------
    ThresholdResult
    """
    assert k > 0.0, f"k must be positive; got {k}."
    assert ai.size > 0, "ai must be non-empty."
    assert np.isfinite(ai).all(), "ai contains non-finite values; filter before calling."
    median = float(np.median(ai))
    mad = float(np.median(np.abs(ai - median)))
    assert mad > 0.0, "MAD is zero; AI distribution is degenerate (all-identical values)."
    threshold = median + k * mad
    return ThresholdResult(threshold=threshold, median=median, mad=mad, k=k)


@dataclass(frozen=True)
class Blob:
    """One spatially-coherent group of AI>threshold pixels.

    Attributes
    ----------
    blob_id      : integer id, ranked by descending `size` (0 = largest).
    size         : number of pixels in this blob.
    centroid_lat, centroid_lon : pixel-mean blob location (degrees).
    hull         : concave hull of the blob's pixels (lon, lat) — follows
        the actual pixel footprint rather than filling in concavities.
    """

    blob_id: int
    size: int
    centroid_lat: float
    centroid_lon: float
    hull: BaseGeometry


@dataclass(frozen=True)
class BlobClustering:
    """Result of clustering AI>threshold pixels into spatial blobs.

    Attributes
    ----------
    labels : (n,) int array, same length/order as the input lat/lon/ai.
        -2 = pixel below threshold; -1 = DBSCAN noise (unclustered
        outlier, always excluded); >=0 = `Blob.blob_id` of the blob this
        pixel belongs to.
    blobs  : blobs found, ranked by descending size (index 0 = largest).
    """

    labels: np.ndarray
    blobs: list[Blob]


def cluster_ai_blobs(
    lat: np.ndarray,
    lon: np.ndarray,
    ai: np.ndarray,
    threshold: float,
    eps_km: float,
    min_samples: int,
    concave_ratio: float,
) -> BlobClustering:
    """Cluster AI>threshold pixels into spatial blobs via DBSCAN (haversine).

    Each above-threshold pixel is assigned to a blob or marked as noise
    (-1) if it doesn't belong to a sufficiently dense group. Blobs are
    ranked by descending pixel count, so the physical plume is typically
    (but not necessarily) blob_id 0; small, spatially separate blobs are
    candidate "noise plumes" to drop via `build_boundary_from_blobs`.

    Parameters
    ----------
    lat, lon, ai : (n,) arrays, identical shape.
    threshold     : AI threshold defining plume-candidate pixels.
    eps_km        : DBSCAN neighborhood radius (km); pixels closer than this
        are considered connected. Should be a few TROPOMI pixel widths
        (nadir pixel is roughly 3.5 x 7 km).
    min_samples   : minimum pixel count for a group to form a blob (DBSCAN
        core-point threshold); smaller groups become noise (-1).
    concave_ratio : `shapely.concave_hull` ratio in [0, 1] used for each
        blob's footprint. 0 hugs the pixel cloud tightly (can get spiky
        with sparse/irregular sampling); 1 is equivalent to the convex
        hull. Tune by eye against the resulting boundary plot.

    Returns
    -------
    BlobClustering
    """
    assert lat.shape == lon.shape == ai.shape, (
        f"Shape mismatch: lat {lat.shape}, lon {lon.shape}, ai {ai.shape}."
    )
    assert eps_km > 0.0, f"eps_km must be positive; got {eps_km}."
    assert min_samples >= 1, f"min_samples must be >= 1; got {min_samples}."
    assert 0.0 <= concave_ratio <= 1.0, f"concave_ratio must be in [0, 1]; got {concave_ratio}."

    above = (ai > threshold) & np.isfinite(ai)
    assert above.any(), f"No pixels above threshold={threshold}."

    above_idx = np.where(above)[0]
    coords_rad = np.radians(np.column_stack([lat[above_idx], lon[above_idx]]))
    raw_labels = DBSCAN(
        eps=eps_km / EARTH_RADIUS_KM, min_samples=min_samples, metric="haversine"
    ).fit_predict(coords_rad)

    labels = np.full(lat.shape, -2, dtype=int)  # -2 = below threshold
    labels[above_idx] = -1  # default to DBSCAN-noise; overwritten below for real blobs

    unique_raw = sorted(set(raw_labels.tolist()) - {-1})
    sizes = {u: int((raw_labels == u).sum()) for u in unique_raw}
    ranked = sorted(unique_raw, key=lambda u: sizes[u], reverse=True)

    blobs = []
    for new_id, raw_id in enumerate(ranked):
        pixel_idx = above_idx[raw_labels == raw_id]
        labels[pixel_idx] = new_id
        blob_lat, blob_lon = lat[pixel_idx], lon[pixel_idx]
        points = MultiPoint(np.column_stack([blob_lon, blob_lat]))
        hull = shapely.concave_hull(points, ratio=concave_ratio)
        if hull.geom_type != "Polygon":
            # Degenerate (collinear/near-collinear) blob: give it a small
            # but non-zero footprint instead of a zero-area line/point.
            hull = hull.buffer(1e-3)
        blobs.append(
            Blob(
                blob_id=new_id,
                size=int(pixel_idx.size),
                centroid_lat=float(blob_lat.mean()),
                centroid_lon=float(blob_lon.mean()),
                hull=hull,
            )
        )

    assert blobs, (
        "No blobs survived DBSCAN clustering (all above-threshold pixels were noise); "
        "lower min_samples or increase eps_km."
    )
    return BlobClustering(labels=labels, blobs=blobs)


def build_boundary_from_blobs(
    clustering: BlobClustering, keep_blob_ids: Sequence[int]
) -> BaseGeometry:
    """Union the hulls of selected blobs into the final inclusion geometry.

    Parameters
    ----------
    clustering    : output of `cluster_ai_blobs`.
    keep_blob_ids : `Blob.blob_id` values to keep as the plume; all other
        blobs (and DBSCAN noise, label -1) are excluded as noise plumes.

    Returns
    -------
    shapely Polygon or MultiPolygon.
    """
    assert len(keep_blob_ids) >= 1, "keep_blob_ids must be non-empty."
    by_id = {b.blob_id: b for b in clustering.blobs}
    missing = sorted(set(keep_blob_ids) - set(by_id))
    assert not missing, f"keep_blob_ids references unknown blob id(s): {missing}."
    geometry = unary_union([by_id[i].hull for i in keep_blob_ids])
    assert geometry.area > 0.0, "Combined boundary has zero area."
    return geometry


def plume_mask(
    lat: np.ndarray,
    lon: np.ndarray,
    ai: np.ndarray,
    boundary: BaseGeometry,
    threshold: float,
) -> np.ndarray:
    """Boolean plume mask: AI > threshold AND inside the inclusion boundary.

    Parameters
    ----------
    lat, lon, ai : (n,) arrays, identical shape.
    boundary     : inclusion geometry (lon, lat), e.g. from
        `build_boundary_from_blobs`.
    threshold    : AI threshold used to derive `boundary`.

    Returns
    -------
    (n,) bool array.
    """
    assert lat.shape == lon.shape == ai.shape, (
        f"Shape mismatch: lat {lat.shape}, lon {lon.shape}, ai {ai.shape}."
    )
    in_boundary = shapely.contains_xy(boundary, lon, lat)
    mask = in_boundary & (ai > threshold) & np.isfinite(ai)
    assert mask.any(), (
        "Plume mask is empty: no pixels satisfy AI > threshold inside the boundary. "
        "Check the threshold and kept blob ids."
    )
    return mask


def _git_hash() -> str:
    """Short git hash of the current HEAD (for provenance)."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def save_plume_boundary(
    path: Path,
    date: str,
    boundary: BaseGeometry,
    threshold_result: ThresholdResult,
    source_files: Sequence[str],
) -> None:
    """Persist a plume boundary and its threshold provenance as GeoJSON.

    Parameters
    ----------
    path             : output .geojson path; parent directory must exist.
    date             : ISO date string, e.g. "2026-01-18".
    boundary         : inclusion geometry (lon, lat); Polygon or MultiPolygon.
    threshold_result : ThresholdResult used to derive the AI threshold.
    source_files     : AER_AI L2 filenames the boundary was drawn against.
    """
    assert path.parent.is_dir(), f"Parent directory does not exist: {path.parent}"
    feature = {
        "type": "Feature",
        "geometry": mapping(boundary),
        "properties": {
            "date": date,
            **asdict(threshold_result),
            "source_files": [str(f) for f in source_files],
            "git_hash": _git_hash(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    with open(path, "w") as f:
        json.dump(feature, f, indent=2)


@dataclass(frozen=True)
class PlumeBoundary:
    """A loaded plume boundary and its full provenance."""

    boundary: BaseGeometry
    threshold: float
    date: str
    median: float
    mad: float
    k: float
    source_files: tuple[str, ...]
    git_hash: str
    created_at: str


def load_plume_boundary(path: Path) -> PlumeBoundary:
    """Load a plume boundary previously written by `save_plume_boundary`."""
    assert path.is_file(), f"Plume boundary file not found: {path}"
    with open(path) as f:
        feature = json.load(f)
    props = feature["properties"]
    return PlumeBoundary(
        boundary=shape(feature["geometry"]),
        threshold=props["threshold"],
        date=props["date"],
        median=props["median"],
        mad=props["mad"],
        k=props["k"],
        source_files=tuple(props["source_files"]),
        git_hash=props["git_hash"],
        created_at=props["created_at"],
    )
