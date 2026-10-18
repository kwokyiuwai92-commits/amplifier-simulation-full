"""Display-space helpers for readable, gain-aligned optical paths.

Pump and seed are independent branches and therefore have independent local
path origins.  The display first translates each branch so its first crystal
hit occupies one shared physical gain volume.  Long upstream free space may
then be compressed without changing the crystal's longitudinal scale.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from .bench_preview import BenchPreview, PreviewElement


@dataclass(frozen=True)
class OpticalPathDisplayScale:
    physical_start_m: float
    first_gain_m: float
    last_gain_m: float
    physical_end_m: float
    upstream_display_m: float
    downstream_display_m: float
    compressed: bool

    def map_m(self, coordinate_m: float | np.ndarray) -> float | np.ndarray:
        values = np.asarray(coordinate_m, dtype=float)
        if not self.compressed:
            mapped = values.copy()
        else:
            mapped = np.empty_like(values)
            upstream = values <= self.first_gain_m
            gain = (values > self.first_gain_m) & (values <= self.last_gain_m)
            downstream = values > self.last_gain_m
            upstream_length = self.first_gain_m - self.physical_start_m
            if upstream_length > 0.0:
                mapped[upstream] = self.first_gain_m - (
                    (self.first_gain_m - values[upstream]) / upstream_length * self.upstream_display_m
                )
            else:
                mapped[upstream] = values[upstream]
            mapped[gain] = values[gain]
            trailing_length = self.physical_end_m - self.last_gain_m
            if trailing_length > 0.0:
                mapped[downstream] = (
                    self.last_gain_m
                    + (values[downstream] - self.last_gain_m) / trailing_length * self.downstream_display_m
                )
            else:
                mapped[downstream] = values[downstream]
        if np.ndim(coordinate_m) == 0:
            return float(mapped)
        return mapped

    @property
    def display_end_m(self) -> float:
        return float(self.map_m(self.physical_end_m))

    @property
    def display_start_m(self) -> float:
        return float(self.map_m(self.physical_start_m))

    def axis_ticks_mm(self) -> tuple[list[float], list[str]]:
        physical = [self.physical_start_m, self.first_gain_m, self.last_gain_m, self.physical_end_m]
        unique: list[float] = []
        for value in physical:
            if not unique or not np.isclose(value, unique[-1], rtol=0.0, atol=1e-12):
                unique.append(value)
        displayed = [float(self.map_m(value)) * 1e3 for value in unique]
        labels = [f"{value * 1e3:.5g}" for value in unique]
        return displayed, labels


@dataclass(frozen=True)
class OpticalTraceAnchor:
    """One solver-recorded D4-sigma plane placed on a preview path."""

    beam_kind: str
    name: str
    optical_path_m: float
    radius_x_D4sigma_m: float
    radius_y_D4sigma_m: float
    centroid_x_m: float
    centroid_y_m: float


def optical_trace_anchors(
    preview: BenchPreview,
    plane_summaries: Sequence[dict[str, Any]],
    gain_path_samples: Sequence[dict[str, Any]] = (),
) -> tuple[OpticalTraceAnchor, ...]:
    """Map authoritative solver planes to their longitudinal bench positions.

    The realtime preview has a continuous cold-state q trace, while the full
    solver records sampled complex fields without storing redundant path
    coordinates in every summary row.  Source, gain surfaces and system output
    have unambiguous positions on the preview, so expose those as hot/result
    anchors instead of pretending that the cold q curve contains thermal-lens,
    gain-guiding or aperture physics.
    """

    beam_kind = str(preview.beam_kind)
    prefix = f"{beam_kind}_"
    gains = sorted(
        (element for element in preview.elements if element.kind == "gain_medium"),
        key=lambda element: (
            int(element.pass_index or 0),
            float(element.start_m),
        ),
    )
    if len(preview.optical_path_m) == 0:
        return ()

    anchors: list[OpticalTraceAnchor] = []
    seen: set[tuple[str, float]] = set()
    for raw in plane_summaries:
        name = str(raw.get("name", ""))
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        coordinate: float | None = None
        if suffix == "source":
            coordinate = float(preview.optical_path_m[0])
        elif suffix == "system_output":
            coordinate = float(preview.optical_path_m[-1])
        elif beam_kind == "pump" and suffix in {"gain_entrance", "gain_exit"} and gains:
            coordinate = float(gains[0].start_m if suffix.endswith("entrance") else gains[0].end_m)
        else:
            match = re.fullmatch(r"pass_(\d+)_gain_(entrance|exit)", suffix)
            if match:
                pass_index = int(match.group(1))
                gain = next(
                    (element for element in gains if int(element.pass_index or 0) == pass_index),
                    None,
                )
                if gain is not None:
                    coordinate = float(
                        gain.start_m if match.group(2) == "entrance" else gain.end_m
                    )
        required = (
            raw.get("radius_x_D4sigma_m"),
            raw.get("radius_y_D4sigma_m"),
            raw.get("centroid_x_m"),
            raw.get("centroid_y_m"),
        )
        if coordinate is None or any(value is None for value in required):
            continue
        key = (name, coordinate)
        if key in seen:
            continue
        seen.add(key)
        anchors.append(
            OpticalTraceAnchor(
                beam_kind=beam_kind,
                name=name,
                optical_path_m=coordinate,
                radius_x_D4sigma_m=float(required[0]),
                radius_y_D4sigma_m=float(required[1]),
                centroid_x_m=float(required[2]),
                centroid_y_m=float(required[3]),
            )
        )
    for raw in gain_path_samples:
        if str(raw.get("beam_kind", "")) != beam_kind:
            continue
        pass_index = int(raw.get("pass_index", 1))
        gain = next(
            (element for element in gains if int(element.pass_index or 0) == pass_index),
            None,
        )
        if gain is None and beam_kind == "pump" and gains and pass_index == 1:
            gain = gains[0]
        relative = raw.get("gain_relative_path_m")
        required = (
            raw.get("radius_x_D4sigma_m"),
            raw.get("radius_y_D4sigma_m"),
            raw.get("centroid_x_m"),
            raw.get("centroid_y_m"),
        )
        if gain is None or relative is None or any(value is None for value in required):
            continue
        relative_path = float(relative)
        gain_length = max(float(gain.end_m - gain.start_m), 0.0)
        if relative_path < -1e-12 or relative_path > gain_length + 1e-12:
            continue
        coordinate = float(gain.start_m) + float(np.clip(relative_path, 0.0, gain_length))
        name = (
            f"{beam_kind}_pass_{pass_index}_gain_zbin_"
            f"{int(raw.get('z_bin_index', len(anchors)))}"
        )
        key = (name, coordinate)
        if key in seen:
            continue
        seen.add(key)
        anchors.append(
            OpticalTraceAnchor(
                beam_kind=beam_kind,
                name=name,
                optical_path_m=coordinate,
                radius_x_D4sigma_m=float(required[0]),
                radius_y_D4sigma_m=float(required[1]),
                centroid_x_m=float(required[2]),
                centroid_y_m=float(required[3]),
            )
        )
    anchors.sort(key=lambda anchor: (anchor.optical_path_m, anchor.name))
    return tuple(anchors)


def build_path_display_scale(
    previews: Sequence[BenchPreview],
    *,
    focus_gain: bool = True,
) -> OpticalPathDisplayScale:
    physical_start = min((float(preview.optical_path_m[0]) for preview in previews if len(preview.optical_path_m)), default=0.0)
    physical_end = max((float(preview.optical_path_m[-1]) for preview in previews if len(preview.optical_path_m)), default=0.0)
    gain_elements = [
        element
        for preview in previews
        for element in preview.elements
        if element.kind == "gain_medium"
    ]
    if not gain_elements:
        return OpticalPathDisplayScale(physical_start, physical_start, physical_end, physical_end, 0.0, 0.0, False)
    first_gain = min(element.start_m for element in gain_elements)
    last_gain = max(element.end_m for element in gain_elements)
    gain_span = max(last_gain - first_gain, max(element.end_m - element.start_m for element in gain_elements), 1e-6)
    trailing = max(physical_end - last_gain, 0.0)
    upstream = max(first_gain - physical_start, 0.0)
    should_compress = bool(focus_gain and upstream > 1.5 * gain_span)
    if not should_compress:
        return OpticalPathDisplayScale(physical_start, first_gain, last_gain, physical_end, upstream, trailing, False)
    return OpticalPathDisplayScale(
        physical_start_m=physical_start,
        first_gain_m=first_gain,
        last_gain_m=last_gain,
        physical_end_m=physical_end,
        upstream_display_m=0.50 * gain_span,
        downstream_display_m=min(trailing, 0.25 * gain_span),
        compressed=True,
    )


def align_previews_to_shared_gain(
    previews: Sequence[BenchPreview],
    crystal_length_m: float,
) -> tuple[BenchPreview, ...]:
    """Place independent Pump/Seed source origins around one shared first gain.

    Each branch keeps its own true cumulative path in the original preview.
    This returned copy is display-only: the first gain is mapped to ``0..L``;
    different source-to-gain distances naturally appear as different negative
    source coordinates, so no user-entered branch offset is required.
    """

    length = float(crystal_length_m)
    if length <= 0.0:
        raise ValueError("crystal_length_m must be positive")
    aligned: list[BenchPreview] = []
    for preview in previews:
        first_gain = next((element for element in preview.elements if element.kind == "gain_medium"), None)
        if first_gain is None:
            aligned.append(preview)
            continue
        local_gain_length = max(first_gain.end_m - first_gain.start_m, np.finfo(float).tiny)

        def transform(values: float | np.ndarray) -> float | np.ndarray:
            source = np.asarray(values, dtype=float)
            target = np.empty_like(source)
            before = source <= first_gain.start_m
            inside = (source > first_gain.start_m) & (source <= first_gain.end_m)
            after = source > first_gain.end_m
            target[before] = source[before] - first_gain.start_m
            target[inside] = (source[inside] - first_gain.start_m) / local_gain_length * length
            target[after] = length + source[after] - first_gain.end_m
            if np.ndim(values) == 0:
                return float(target)
            return target

        elements: list[PreviewElement] = []
        for element in preview.elements:
            if element is first_gain:
                start_m, end_m = 0.0, length
            else:
                start_m = float(transform(element.start_m))
                end_m = float(transform(element.end_m))
            elements.append(
                replace(
                    element,
                    start_m=start_m,
                    end_m=end_m,
                )
            )
        aligned.append(
            BenchPreview(
                beam_kind=preview.beam_kind,
                optical_path_m=np.asarray(transform(preview.optical_path_m), dtype=float),
                radius_x_D4sigma_m=preview.radius_x_D4sigma_m,
                radius_y_D4sigma_m=preview.radius_y_D4sigma_m,
                centroid_x_m=preview.centroid_x_m,
                centroid_y_m=preview.centroid_y_m,
                elements=tuple(elements),
                warnings=preview.warnings,
            )
        )
    return tuple(aligned)


def gain_sample_mask(preview: BenchPreview) -> np.ndarray:
    """Select samples physically located inside one of the preview's gain hits."""

    coordinate = np.asarray(preview.optical_path_m, dtype=float)
    mask = np.zeros(len(coordinate), dtype=bool)
    tolerance = 1e-12 * max(
        float(np.max(np.abs(coordinate))) if len(coordinate) else 0.0,
        float(np.ptp(coordinate)) if len(coordinate) else 0.0,
        np.finfo(float).tiny,
    )
    for element in preview.elements:
        if element.kind == "gain_medium":
            mask |= (coordinate >= element.start_m - tolerance) & (coordinate <= element.end_m + tolerance)
    return mask
