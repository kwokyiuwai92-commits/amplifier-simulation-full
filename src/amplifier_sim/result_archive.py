"""Portable, reloadable result archives for single runs and parameter sweeps."""

from __future__ import annotations

import hashlib
import hmac
import io
import pickle
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Iterable
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile

import numpy as np

from .grid import Grid3D
from .json_contract import dumps_json_portable, loads_json_strict
from .result_fields import (
    OPTIONAL_RESULT_ARRAYS,
    REQUIRED_RESULT_ARRAYS,
    standard_result_arrays,
)
from .strict_validation import NON_QUANTITATIVE_BANNER

if TYPE_CHECKING:
    from .solver import SimulationResult


ARCHIVE_FORMAT = "amplifier-sim-result"
ARCHIVE_SCHEMA_VERSION = 1


class ResultArchiveError(ValueError):
    """Raised when an uploaded result archive is missing or inconsistent."""


def _require_json_integer(
    value: Any,
    *,
    label: str,
    minimum: int = 0,
) -> int:
    """Accept JSON integer tokens only; never truncate floats or booleans."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ResultArchiveError(f"{label} must be an integer")
    if value < minimum:
        raise ResultArchiveError(f"{label} must be at least {minimum}")
    return value


@dataclass(frozen=True)
class ArchiveResourceBudget:
    """Fail-closed resource limits for untrusted result archives."""

    maximum_archive_bytes: int = 1 * 1024**3
    maximum_member_count: int = 4096
    maximum_total_uncompressed_bytes: int = 4 * 1024**3
    maximum_member_uncompressed_bytes: int = 2 * 1024**3
    maximum_compression_ratio: float = 10_000.0
    maximum_json_bytes: int = 64 * 1024**2
    maximum_npz_bytes: int = 2 * 1024**3
    maximum_array_count: int = 4096
    maximum_total_array_bytes: int = 4 * 1024**3
    maximum_array_bytes: int = 2 * 1024**3
    maximum_array_elements: int = 500_000_000
    maximum_array_dimensions: int = 8
    maximum_axis_length: int = 10_000_000
    maximum_npy_header_bytes: int = 64 * 1024


@dataclass
class _ArchiveReadUsage:
    array_count: int = 0
    array_bytes: int = 0
    array_elements: int = 0


@dataclass(frozen=True)
class ArchivePoint:
    """One independently executed operating point in a result archive."""

    record: dict[str, Any]
    config: dict[str, Any]
    result: SimulationResult | None = None


def require_completed_sweep_results(
    points: Iterable[ArchivePoint],
) -> tuple[ArchivePoint, ...]:
    """Reject a sweep archive that would silently omit completed results.

    Failed points legitimately have no ``SimulationResult`` and remain useful
    diagnostic records.  Every other sweep status represents a completed
    solve, so exporting it without its arrays would turn a full-result download
    into an unlabelled summary-only archive.
    """

    point_list = tuple(points)
    missing = [
        index + 1
        for index, point in enumerate(point_list)
        if str(point.record.get("status", "")) != "execution_failed"
        and point.result is None
    ]
    if missing:
        labels = ", ".join(str(index) for index in missing)
        raise ResultArchiveError(
            "completed sweep results are unavailable for point(s) "
            f"{labels}; refusing to create a summary-only fallback archive"
        )
    return point_list


@dataclass(frozen=True)
class ArchivePointFile:
    """One trusted local point whose full result is loaded only while writing."""

    record: dict[str, Any]
    config: dict[str, Any]
    result_pickle_path: str | PathLike[str] | None = None


@dataclass(frozen=True)
class ResultArchive:
    """Decoded archive metadata and all of its operating points."""

    manifest: dict[str, Any]
    points: tuple[ArchivePoint, ...]
    base_config: dict[str, Any] | None = None


ArchiveSource = str | PathLike[str] | BinaryIO


def python_warning_record(warning: Any) -> dict[str, Any]:
    """Convert one ``warnings.catch_warnings`` item into portable metadata."""

    return {
        "category": warning.category.__name__,
        "message": str(warning.message),
        "filename": str(warning.filename),
        "line": int(warning.lineno),
    }


def diagnostic_record_for_result(
    result: SimulationResult,
    *,
    python_warnings: Iterable[dict[str, Any]] = (),
    **extra: Any,
) -> dict[str, Any]:
    """Build the shared status and notice record for one completed result."""

    validation = result.summary.get("validation", {})
    failures = list(validation.get("failures", []))
    validation_warnings = list(validation.get("warnings", []))
    if not bool(validation.get("valid", True)):
        status = "execution_completed_but_invalid"
    elif validation_warnings:
        status = "valid_with_warnings"
    else:
        status = "valid"
    signal = result.summary.get("signal", {})
    requested_model = str(
        signal.get("requested_model", signal.get("model", "unknown"))
    )
    executed_model = str(
        signal.get("executed_model", signal.get("model", "unknown"))
    )
    return {
        **extra,
        "status": status,
        "error": None,
        "error_type": None,
        "error_message": None,
        "traceback": None,
        "validation_mode": str(validation.get("mode", "exploratory")),
        "validation_enforced": bool(validation.get("enforced", False)),
        "validation_status": str(
            validation.get(
                "status",
                "valid" if validation.get("valid", True) else "invalid",
            )
        ),
        "invalid": not bool(validation.get("valid", True)),
        "quantitative": bool(validation.get("valid", True)),
        "quantitative_scope": validation.get("quantitative_scope"),
        "banner": (
            None
            if bool(validation.get("valid", True))
            else NON_QUANTITATIVE_BANNER
        ),
        "validation_failures": failures,
        "validation_warnings": validation_warnings,
        "solver_warnings": list(result.summary.get("warnings", [])),
        "python_warnings": list(python_warnings),
        "requested_signal_model": requested_model,
        "executed_signal_model": executed_model,
        "signal_fallback_active": bool(
            signal.get("fallback_active", executed_model != requested_model)
        ),
    }


def build_result_archive(
    points: Iterable[ArchivePoint],
    *,
    run_kind: str,
    base_config: dict[str, Any] | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> bytes:
    """Serialize every result, configuration, warning and failure into one ZIP."""

    payload = io.BytesIO()
    write_result_archive(
        payload,
        points,
        run_kind=run_kind,
        base_config=base_config,
        run_metadata=run_metadata,
    )
    return payload.getvalue()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_npz_path(
    path: Path,
    arrays: dict[str, np.ndarray],
    *,
    compression_level: int,
) -> None:
    """Write one portable NPZ without materializing the NPZ as bytes."""

    with ZipFile(
        path,
        "w",
        compression=ZIP_DEFLATED,
        compresslevel=compression_level,
        allowZip64=True,
    ) as archive:
        for name, values in arrays.items():
            array = np.asanyarray(values)
            with archive.open(f"{name}.npy", "w", force_zip64=True) as stream:
                # Preserve the previous trusted-export behavior for diagnostic
                # object arrays. The untrusted loader still rejects them before
                # deserialization under its dtype resource contract.
                np.lib.format.write_array(
                    stream,
                    array,
                    allow_pickle=bool(array.dtype.hasobject),
                )


def _write_result_disk_backed(
    archive: ZipFile,
    prefix: str,
    result: SimulationResult,
    *,
    staging_directory: Path,
    compression_level: int,
) -> None:
    """Serialize one point while keeping its nested NPZ payloads off heap."""

    _write_json(archive, f"{prefix}/summary.json", result.summary)
    arrays = standard_result_arrays(result)
    metadata: dict[str, Any] = {
        "level_names": list(result.level_names),
        "cpa_diagnostics": result.cpa_diagnostics,
        "optional_arrays": [
            name for name in OPTIONAL_RESULT_ARRAYS if name in arrays
        ],
        "stress_arrays": [],
        "has_optical_grid": result.optical_grid is not None,
        "has_pump_optical_grid": result.pump_optical_grid is not None,
        "optical_planes": [],
    }
    for index, (name, values) in enumerate(result.stress.items()):
        key = f"stress_{index:04d}"
        arrays[key] = np.asarray(values)
        metadata["stress_arrays"].append({"name": name, "key": key})

    if result.optical_planes:
        plane_arrays: dict[str, np.ndarray] = {}
        coordinates = result.optical_plane_coordinates or {}
        for index, (name, values) in enumerate(result.optical_planes.items()):
            key = f"plane_{index:04d}"
            plane_arrays[key] = np.asarray(values)
            coordinate_keys: list[str] | None = None
            if name in coordinates:
                x_key = f"{key}_x_m"
                y_key = f"{key}_y_m"
                arrays[x_key] = np.asarray(coordinates[name][0])
                arrays[y_key] = np.asarray(coordinates[name][1])
                coordinate_keys = [x_key, y_key]
            metadata["optical_planes"].append(
                {"name": name, "key": key, "coordinate_keys": coordinate_keys}
            )
        with tempfile.NamedTemporaryFile(
            prefix="amplifier-planes-",
            suffix=".npz",
            dir=staging_directory,
            delete=False,
        ) as temporary:
            plane_path = Path(temporary.name)
        try:
            _write_npz_path(
                plane_path,
                plane_arrays,
                compression_level=compression_level,
            )
            metadata["optical_planes_sha256"] = _sha256_path(plane_path)
            archive.write(
                plane_path,
                f"{prefix}/optical_planes.npz",
                compress_type=ZIP_STORED,
            )
        finally:
            plane_path.unlink(missing_ok=True)

    with tempfile.NamedTemporaryFile(
        prefix="amplifier-fields-",
        suffix=".npz",
        dir=staging_directory,
        delete=False,
    ) as temporary:
        field_path = Path(temporary.name)
    try:
        _write_npz_path(
            field_path,
            arrays,
            compression_level=compression_level,
        )
        metadata["fields_sha256"] = _sha256_path(field_path)
        archive.write(
            field_path,
            f"{prefix}/fields.npz",
            compress_type=ZIP_STORED,
        )
    finally:
        field_path.unlink(missing_ok=True)
    _write_json(archive, f"{prefix}/result_metadata.json", metadata)


def write_result_archive(
    destination: BinaryIO,
    points: Iterable[ArchivePoint],
    *,
    run_kind: str,
    base_config: dict[str, Any] | None = None,
    run_metadata: dict[str, Any] | None = None,
    staging_parent: str | PathLike[str] | None = None,
    compression_level: int = 6,
) -> None:
    """Stream an archive to a file-like destination using bounded disk staging.

    No final archive is retained by this function.  The temporary directory and
    every per-point NPZ are removed on normal completion and on any exception,
    including a browser disconnect propagated by the destination stream.
    """

    point_list = tuple(points)
    if not point_list:
        raise ValueError("a result archive must contain at least one point")

    public_records = [
        _public_record(point.record, point.result is not None)
        for point in point_list
    ]
    result_available = [point.result is not None for point in point_list]
    manifest = _build_result_archive_manifest(
        public_records,
        result_available,
        run_kind=run_kind,
        run_metadata=run_metadata,
    )

    with tempfile.TemporaryDirectory(
        prefix="amplifier-archive-",
        dir=staging_parent,
    ) as temporary_directory:
        staging_directory = Path(temporary_directory)
        with ZipFile(destination, "w", allowZip64=True) as archive:
            _write_json(archive, "manifest.json", manifest)
            if base_config is not None:
                _write_json(archive, "base_config.json", base_config)
            for index, (point, public_record) in enumerate(
                zip(point_list, public_records, strict=True)
            ):
                prefix = f"points/{index:04d}"
                _write_json(archive, f"{prefix}/record.json", public_record)
                _write_json(archive, f"{prefix}/config.json", point.config)
                if point.result is not None:
                    _write_result_disk_backed(
                        archive,
                        prefix,
                        point.result,
                        staging_directory=staging_directory,
                        compression_level=compression_level,
                    )


def build_result_archive_from_point_files(
    points: Iterable[ArchivePointFile],
    *,
    run_kind: str,
    base_config: dict[str, Any] | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> bytes:
    """Build a full archive while loading at most one point result at a time."""

    payload = io.BytesIO()
    write_result_archive_from_point_files(
        payload,
        points,
        run_kind=run_kind,
        base_config=base_config,
        run_metadata=run_metadata,
    )
    return payload.getvalue()


def write_result_archive_from_point_files(
    destination: BinaryIO,
    points: Iterable[ArchivePointFile],
    *,
    run_kind: str,
    base_config: dict[str, Any] | None = None,
    run_metadata: dict[str, Any] | None = None,
    staging_parent: str | PathLike[str] | None = None,
    compression_level: int = 6,
) -> None:
    """Stream trusted persisted points without retaining the completed ZIP."""

    point_list = tuple(points)
    if not point_list:
        raise ValueError("a result archive must contain at least one point")
    result_available = [
        point.result_pickle_path is not None for point in point_list
    ]
    public_records = [
        _public_record(point.record, available)
        for point, available in zip(point_list, result_available, strict=True)
    ]
    manifest = _build_result_archive_manifest(
        public_records,
        result_available,
        run_kind=run_kind,
        run_metadata=run_metadata,
    )

    with tempfile.TemporaryDirectory(
        prefix="amplifier-archive-",
        dir=staging_parent,
    ) as temporary_directory:
        staging_directory = Path(temporary_directory)
        with ZipFile(destination, "w", allowZip64=True) as archive:
            _write_json(archive, "manifest.json", manifest)
            if base_config is not None:
                _write_json(archive, "base_config.json", base_config)
            for index, (point, public_record, available) in enumerate(
                zip(point_list, public_records, result_available, strict=True)
            ):
                prefix = f"points/{index:04d}"
                _write_json(archive, f"{prefix}/record.json", public_record)
                _write_json(archive, f"{prefix}/config.json", point.config)
                if not available:
                    continue
                result_path = Path(point.result_pickle_path)
                try:
                    with result_path.open("rb") as stream:
                        result = pickle.load(stream)
                except Exception as exc:
                    raise ResultArchiveError(
                        f"could not load trusted point result {result_path}: {exc}"
                    ) from exc
                try:
                    _write_result_disk_backed(
                        archive,
                        prefix,
                        result,
                        staging_directory=staging_directory,
                        compression_level=compression_level,
                    )
                finally:
                    del result


def _build_result_archive_manifest(
    public_records: list[dict[str, Any]],
    result_available: list[bool],
    *,
    run_kind: str,
    run_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    status_counts = Counter(
        str(record.get("status", "unknown")) for record in public_records
    )
    invalid_count = sum(
        bool(record.get("invalid", False))
        or str(record.get("status", "")) == "execution_completed_but_invalid"
        for record in public_records
    )
    execution_error_count = sum(
        bool(record.get("error")) for record in public_records
    )
    notice_counts = {
        "execution_errors": sum(
            bool(record.get("error")) for record in public_records
        ),
        "solver_warnings": sum(
            len(record.get("solver_warnings", [])) for record in public_records
        ),
        "python_warnings": sum(
            len(record.get("python_warnings", [])) for record in public_records
        ),
        "validation_failures": sum(
            len(record.get("validation_failures", []))
            for record in public_records
        ),
        "validation_warnings": sum(
            len(record.get("validation_warnings", []))
            for record in public_records
        ),
    }
    return {
        "format": ARCHIVE_FORMAT,
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_kind": str(run_kind),
        "point_count": len(public_records),
        "completed_result_count": sum(result_available),
        "status_counts": dict(status_counts),
        "quantitative_status": (
            "non_quantitative_points_present"
            if invalid_count
            else "incomplete_execution"
            if execution_error_count
            else "validation_contracts_passed"
        ),
        "non_quantitative_point_count": invalid_count,
        "banner": NON_QUANTITATIVE_BANNER if invalid_count else None,
        "notice_counts": notice_counts,
        "run_metadata": dict(run_metadata or {}),
        "points": [
            {
                "id": f"{index:04d}",
                "status": record.get("status"),
                "result_available": bool(available),
            }
            for index, (record, available) in enumerate(
                zip(public_records, result_available, strict=True)
            )
        ],
    }


def load_result_archive(
    data: bytes | bytearray | memoryview,
    *,
    resource_budget: ArchiveResourceBudget | None = None,
) -> ResultArchive:
    """Decode an archive produced by :func:`build_result_archive`."""

    budget = resource_budget or ArchiveResourceBudget()
    _validate_resource_budget(budget)
    data_size = len(data)
    if data_size > budget.maximum_archive_bytes:
        raise ResultArchiveError(
            "result archive exceeds maximum_archive_bytes "
            f"({data_size} > {budget.maximum_archive_bytes})"
        )
    # Ordinary callers and Streamlit uploads already supply immutable bytes.
    # Reuse that object instead of briefly doubling a potentially large file.
    raw_archive = data if isinstance(data, bytes) else bytes(data)
    usage = _ArchiveReadUsage()
    try:
        with ZipFile(io.BytesIO(raw_archive), "r") as archive:
            _preflight_zip_members(archive, budget, label="result archive")
            manifest, point_entries = _read_archive_manifest(archive, budget)
            base_config = (
                _read_json(archive, "base_config.json", budget)
                if "base_config.json" in archive.namelist()
                else None
            )

            points: list[ArchivePoint] = []
            seen_point_ids: set[str] = set()
            for point_index, point_entry in enumerate(point_entries):
                if not isinstance(point_entry, dict):
                    raise ResultArchiveError(
                        f"manifest.points[{point_index}] must be a JSON object"
                    )
                point_id = str(point_entry.get("id", ""))
                if not point_id.isdigit() or len(point_id) != 4:
                    raise ResultArchiveError(f"invalid point id {point_id!r}")
                if point_id in seen_point_ids:
                    raise ResultArchiveError(f"duplicate point id {point_id!r}")
                seen_point_ids.add(point_id)
                prefix = f"points/{point_id}"
                record = _read_json(archive, f"{prefix}/record.json", budget)
                config = _read_json(archive, f"{prefix}/config.json", budget)
                record_available = record.get("result_available", False)
                entry_available = point_entry.get("result_available", False)
                if not isinstance(record_available, bool) or not isinstance(
                    entry_available, bool
                ):
                    raise ResultArchiveError(
                        f"point {point_id} result_available must be boolean"
                    )
                result_available = record_available
                if result_available != entry_available:
                    raise ResultArchiveError(
                        f"point {point_id} result availability is inconsistent"
                    )
                if str(record.get("status")) != str(point_entry.get("status")):
                    raise ResultArchiveError(
                        f"point {point_id} status is inconsistent between manifest and record"
                    )
                result = (
                    _read_result(archive, prefix, budget, usage)
                    if result_available
                    else None
                )
                points.append(ArchivePoint(record=record, config=config, result=result))
            completed_count = sum(point.result is not None for point in points)
            archived_completed_count = _require_json_integer(
                manifest.get("completed_result_count"),
                label="manifest completed_result_count",
            )
            if archived_completed_count != completed_count:
                raise ResultArchiveError(
                    "manifest completed_result_count does not match decoded results"
                )
            expected_status_counts = Counter(
                str(point.record.get("status", "unknown")) for point in points
            )
            archived_status_counts = manifest.get("status_counts", {})
            try:
                normalized_status_counts = {
                    str(key): _require_json_integer(
                        value,
                        label=f"manifest status_counts[{str(key)!r}]",
                    )
                    for key, value in archived_status_counts.items()
                }
            except (AttributeError, TypeError, ValueError) as exc:
                raise ResultArchiveError(
                    "manifest status_counts must map statuses to integer counts"
                ) from exc
            if normalized_status_counts != dict(expected_status_counts):
                raise ResultArchiveError(
                    "manifest status_counts does not match point records"
                )
    except BadZipFile as exc:
        raise ResultArchiveError("the uploaded file is not a readable ZIP archive") from exc
    except KeyError as exc:
        raise ResultArchiveError(f"archive is missing {exc.args[0]}") from exc
    return ResultArchive(
        manifest=manifest,
        points=tuple(points),
        base_config=base_config,
    )


def load_result_archive_index(
    source: ArchiveSource,
    *,
    resource_budget: ArchiveResourceBudget | None = None,
) -> ResultArchive:
    """Load sweep metadata and point records without decoding result arrays.

    This is the result-analysis path for large sweep archives.  It validates
    the complete outer ZIP contract, but reads only JSON records.  Individual
    fields can subsequently be restored with :func:`load_result_archive_point`.
    """

    budget = resource_budget or ArchiveResourceBudget()
    _validate_resource_budget(budget)
    source_size = _archive_source_size(source)
    if source_size > budget.maximum_archive_bytes:
        raise ResultArchiveError(
            "result archive exceeds maximum_archive_bytes "
            f"({source_size} > {budget.maximum_archive_bytes})"
        )
    try:
        with ZipFile(source, "r") as archive:
            _preflight_zip_members(archive, budget, label="result archive")
            manifest, point_entries = _read_archive_manifest(archive, budget)
            base_config = (
                _read_json(archive, "base_config.json", budget)
                if "base_config.json" in archive.namelist()
                else None
            )
            points: list[ArchivePoint] = []
            seen_point_ids: set[str] = set()
            for point_index, point_entry in enumerate(point_entries):
                point_id = _validated_point_id(
                    point_entry,
                    point_index=point_index,
                    seen_point_ids=seen_point_ids,
                )
                prefix = f"points/{point_id}"
                record = _read_json(archive, f"{prefix}/record.json", budget)
                _validate_point_record(record, point_entry, point_id=point_id)
                # Configs and field arrays are intentionally deferred until a
                # user selects this operating point in the analysis UI.
                points.append(ArchivePoint(record=record, config={}, result=None))
            _validate_index_counts(manifest, point_entries, points)
    except BadZipFile as exc:
        raise ResultArchiveError("the uploaded file is not a readable ZIP archive") from exc
    except KeyError as exc:
        raise ResultArchiveError(f"archive is missing {exc.args[0]}") from exc
    finally:
        _rewind_archive_source(source)
    return ResultArchive(
        manifest=manifest,
        points=tuple(points),
        base_config=base_config,
    )


def load_result_archive_point(
    source: ArchiveSource,
    point_index: int,
    *,
    resource_budget: ArchiveResourceBudget | None = None,
) -> ArchivePoint:
    """Restore one selected point from a previously indexed result archive."""

    budget = resource_budget or ArchiveResourceBudget()
    _validate_resource_budget(budget)
    source_size = _archive_source_size(source)
    if source_size > budget.maximum_archive_bytes:
        raise ResultArchiveError(
            "result archive exceeds maximum_archive_bytes "
            f"({source_size} > {budget.maximum_archive_bytes})"
        )
    try:
        with ZipFile(source, "r") as archive:
            _preflight_zip_members(archive, budget, label="result archive")
            _manifest, point_entries = _read_archive_manifest(archive, budget)
            if point_index < 0 or point_index >= len(point_entries):
                raise ResultArchiveError(
                    f"point index {point_index} is outside the archived range"
                )
            # The point loader is public and may be called without a prior
            # index load.  Validate the complete ID namespace first so a
            # duplicate manifest entry cannot alias another point directory.
            seen_point_ids: set[str] = set()
            validated_point_ids = [
                _validated_point_id(
                    entry,
                    point_index=index,
                    seen_point_ids=seen_point_ids,
                )
                for index, entry in enumerate(point_entries)
            ]
            point_entry = point_entries[point_index]
            point_id = validated_point_ids[point_index]
            prefix = f"points/{point_id}"
            record = _read_json(archive, f"{prefix}/record.json", budget)
            _validate_point_record(record, point_entry, point_id=point_id)
            config = _read_json(archive, f"{prefix}/config.json", budget)
            usage = _ArchiveReadUsage()
            result = (
                _read_result(archive, prefix, budget, usage)
                if bool(point_entry.get("result_available", False))
                else None
            )
    except BadZipFile as exc:
        raise ResultArchiveError("the uploaded file is not a readable ZIP archive") from exc
    except KeyError as exc:
        raise ResultArchiveError(f"archive is missing {exc.args[0]}") from exc
    finally:
        _rewind_archive_source(source)
    return ArchivePoint(record=record, config=config, result=result)


def _archive_source_size(source: ArchiveSource) -> int:
    if isinstance(source, (str, PathLike)):
        return int(Path(source).stat().st_size)
    try:
        position = source.tell()
        source.seek(0, io.SEEK_END)
        size = int(source.tell())
        source.seek(position)
    except (AttributeError, OSError, ValueError) as exc:
        raise ResultArchiveError("result archive source must be seekable") from exc
    return size


def _rewind_archive_source(source: ArchiveSource) -> None:
    if isinstance(source, (str, PathLike)):
        return
    try:
        source.seek(0)
    except (AttributeError, OSError, ValueError):
        pass


def _read_archive_manifest(
    archive: ZipFile,
    budget: ArchiveResourceBudget,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _read_json(archive, "manifest.json", budget)
    if manifest.get("format") != ARCHIVE_FORMAT:
        raise ResultArchiveError("the uploaded ZIP is not an amplifier result archive")
    schema_version = _require_json_integer(
        manifest.get("schema_version"),
        label="manifest schema_version",
    )
    if schema_version != ARCHIVE_SCHEMA_VERSION:
        raise ResultArchiveError(
            "unsupported result archive schema version "
            f"{manifest.get('schema_version')!r}"
        )
    raw_entries = manifest.get("points")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ResultArchiveError("manifest.points must be a non-empty list")
    if any(not isinstance(entry, dict) for entry in raw_entries):
        raise ResultArchiveError("every manifest point must be a JSON object")
    point_entries = list(raw_entries)
    point_count = _require_json_integer(
        manifest.get("point_count"),
        label="manifest point_count",
        minimum=1,
    )
    if point_count != len(point_entries):
        raise ResultArchiveError("manifest point_count does not match manifest.points")
    _require_json_integer(
        manifest.get("completed_result_count"),
        label="manifest completed_result_count",
    )
    if "non_quantitative_point_count" in manifest:
        _require_json_integer(
            manifest["non_quantitative_point_count"],
            label="manifest non_quantitative_point_count",
        )
    notice_counts = manifest.get("notice_counts", {})
    if not isinstance(notice_counts, dict):
        raise ResultArchiveError("manifest notice_counts must be an object")
    for key, value in notice_counts.items():
        _require_json_integer(
            value,
            label=f"manifest notice_counts[{str(key)!r}]",
        )
    return manifest, point_entries


def _validated_point_id(
    point_entry: dict[str, Any],
    *,
    point_index: int,
    seen_point_ids: set[str],
) -> str:
    point_id = str(point_entry.get("id", ""))
    if not point_id.isdigit() or len(point_id) != 4:
        raise ResultArchiveError(
            f"invalid point id {point_id!r} at manifest index {point_index}"
        )
    if point_id in seen_point_ids:
        raise ResultArchiveError(f"duplicate point id {point_id!r}")
    seen_point_ids.add(point_id)
    return point_id


def _validate_point_record(
    record: dict[str, Any],
    point_entry: dict[str, Any],
    *,
    point_id: str,
) -> None:
    record_available = record.get("result_available", False)
    entry_available = point_entry.get("result_available", False)
    if not isinstance(record_available, bool) or not isinstance(entry_available, bool):
        raise ResultArchiveError(
            f"point {point_id} result_available must be boolean"
        )
    if record_available != entry_available:
        raise ResultArchiveError(
            f"point {point_id} result availability is inconsistent"
        )
    if str(record.get("status")) != str(point_entry.get("status")):
        raise ResultArchiveError(
            f"point {point_id} status is inconsistent between manifest and record"
        )


def _validate_index_counts(
    manifest: dict[str, Any],
    point_entries: list[dict[str, Any]],
    points: list[ArchivePoint],
) -> None:
    completed_count = sum(
        bool(entry.get("result_available", False)) for entry in point_entries
    )
    archived_completed_count = _require_json_integer(
        manifest.get("completed_result_count"),
        label="manifest completed_result_count",
    )
    if archived_completed_count != completed_count:
        raise ResultArchiveError(
            "manifest completed_result_count does not match point availability"
        )
    expected_status_counts = Counter(
        str(point.record.get("status", "unknown")) for point in points
    )
    archived_status_counts = manifest.get("status_counts", {})
    try:
        normalized_status_counts = {
            str(key): _require_json_integer(
                value,
                label=f"manifest status_counts[{str(key)!r}]",
            )
            for key, value in archived_status_counts.items()
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise ResultArchiveError(
            "manifest status_counts must map statuses to integer counts"
        ) from exc
    if normalized_status_counts != dict(expected_status_counts):
        raise ResultArchiveError(
            "manifest status_counts does not match point records"
        )


def _validate_resource_budget(budget: ArchiveResourceBudget) -> None:
    integer_limits = {
        name: int(getattr(budget, name))
        for name in (
            "maximum_archive_bytes",
            "maximum_member_count",
            "maximum_total_uncompressed_bytes",
            "maximum_member_uncompressed_bytes",
            "maximum_json_bytes",
            "maximum_npz_bytes",
            "maximum_array_count",
            "maximum_total_array_bytes",
            "maximum_array_bytes",
            "maximum_array_elements",
            "maximum_array_dimensions",
            "maximum_axis_length",
            "maximum_npy_header_bytes",
        )
    }
    if any(value <= 0 for value in integer_limits.values()):
        raise ValueError("archive resource-budget integer limits must be positive")
    ratio = float(budget.maximum_compression_ratio)
    if not math.isfinite(ratio) or ratio < 1.0:
        raise ValueError(
            "archive maximum_compression_ratio must be finite and at least one"
        )


def _preflight_zip_members(
    archive: ZipFile,
    budget: ArchiveResourceBudget,
    *,
    label: str,
) -> None:
    members = archive.infolist()
    if len(members) > budget.maximum_member_count:
        raise ResultArchiveError(
            f"{label} contains too many members "
            f"({len(members)} > {budget.maximum_member_count})"
        )
    names = [item.filename for item in members]
    if len(set(names)) != len(names):
        raise ResultArchiveError(f"{label} contains duplicate member names")
    total_uncompressed = 0
    for item in members:
        name = item.filename
        if (
            not name
            or "\x00" in name
            or "\\" in name
            or name.startswith("/")
            or any(part == ".." for part in name.split("/"))
        ):
            raise ResultArchiveError(f"{label} contains unsafe member name {name!r}")
        if item.flag_bits & 0x1:
            raise ResultArchiveError(f"{label} contains encrypted member {name!r}")
        if item.compress_type not in {ZIP_STORED, ZIP_DEFLATED}:
            raise ResultArchiveError(
                f"{label} member {name!r} uses an unsupported compression method"
            )
        if item.file_size < 0 or item.compress_size < 0:
            raise ResultArchiveError(f"{label} contains an invalid member size")
        if item.file_size > budget.maximum_member_uncompressed_bytes:
            raise ResultArchiveError(
                f"{label} member {name!r} exceeds the uncompressed size limit"
            )
        total_uncompressed += item.file_size
        if total_uncompressed > budget.maximum_total_uncompressed_bytes:
            raise ResultArchiveError(
                f"{label} exceeds the total uncompressed size limit"
            )
        ratio = item.file_size / max(item.compress_size, 1)
        if ratio > budget.maximum_compression_ratio:
            raise ResultArchiveError(
                f"{label} member {name!r} exceeds the compression-ratio limit"
            )


def _public_record(record: dict[str, Any], result_available: bool) -> dict[str, Any]:
    return {
        **{
            str(key): value
            for key, value in record.items()
            if not str(key).startswith("_")
        },
        "result_available": bool(result_available),
    }


def _write_json(archive: ZipFile, name: str, value: Any) -> None:
    archive.writestr(
        name,
        dumps_json_portable(value, ensure_ascii=False, indent=2).encode("utf-8"),
        compress_type=ZIP_DEFLATED,
        compresslevel=6,
    )


def _read_json(
    archive: ZipFile,
    name: str,
    budget: ArchiveResourceBudget,
) -> dict[str, Any]:
    info = archive.getinfo(name)
    if info.file_size > budget.maximum_json_bytes:
        raise ResultArchiveError(
            f"{name} exceeds the JSON resource limit "
            f"({info.file_size} > {budget.maximum_json_bytes})"
        )
    try:
        value = loads_json_strict(archive.read(name).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ResultArchiveError(f"{name} is not valid strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ResultArchiveError(f"{name} must contain a JSON object")
    return value


def _write_result(archive: ZipFile, prefix: str, result: SimulationResult) -> None:
    _write_json(archive, f"{prefix}/summary.json", result.summary)
    arrays = standard_result_arrays(result)
    metadata: dict[str, Any] = {
        "level_names": list(result.level_names),
        "cpa_diagnostics": result.cpa_diagnostics,
        "optional_arrays": [
            name for name in OPTIONAL_RESULT_ARRAYS if name in arrays
        ],
        "stress_arrays": [],
        "has_optical_grid": result.optical_grid is not None,
        "has_pump_optical_grid": result.pump_optical_grid is not None,
        "optical_planes": [],
    }
    for index, (name, values) in enumerate(result.stress.items()):
        key = f"stress_{index:04d}"
        arrays[key] = np.asarray(values)
        metadata["stress_arrays"].append({"name": name, "key": key})
    if result.optical_planes:
        plane_arrays: dict[str, np.ndarray] = {}
        coordinates = result.optical_plane_coordinates or {}
        for index, (name, values) in enumerate(result.optical_planes.items()):
            key = f"plane_{index:04d}"
            plane_arrays[key] = np.asarray(values)
            coordinate_keys: list[str] | None = None
            if name in coordinates:
                x_key = f"{key}_x_m"
                y_key = f"{key}_y_m"
                arrays[x_key] = np.asarray(coordinates[name][0])
                arrays[y_key] = np.asarray(coordinates[name][1])
                coordinate_keys = [x_key, y_key]
            metadata["optical_planes"].append(
                {"name": name, "key": key, "coordinate_keys": coordinate_keys}
            )
        plane_buffer = io.BytesIO()
        np.savez_compressed(plane_buffer, **plane_arrays)
        plane_payload = plane_buffer.getvalue()
        metadata["optical_planes_sha256"] = hashlib.sha256(
            plane_payload
        ).hexdigest()
        archive.writestr(
            f"{prefix}/optical_planes.npz",
            plane_payload,
            compress_type=ZIP_STORED,
        )

    field_buffer = io.BytesIO()
    np.savez_compressed(field_buffer, **arrays)
    field_payload = field_buffer.getvalue()
    metadata["fields_sha256"] = hashlib.sha256(field_payload).hexdigest()
    archive.writestr(
        f"{prefix}/fields.npz",
        field_payload,
        compress_type=ZIP_STORED,
    )
    _write_json(archive, f"{prefix}/result_metadata.json", metadata)


def _inspect_npz(
    data: bytes,
    budget: ArchiveResourceBudget,
    usage: _ArchiveReadUsage,
    *,
    label: str,
) -> None:
    """Read only NPY headers and reject oversized/dangerous payloads."""

    if len(data) > budget.maximum_npz_bytes:
        raise ResultArchiveError(
            f"{label} exceeds maximum_npz_bytes "
            f"({len(data)} > {budget.maximum_npz_bytes})"
        )
    try:
        with ZipFile(io.BytesIO(data), "r") as payload:
            _preflight_zip_members(payload, budget, label=label)
            array_names: set[str] = set()
            for info in payload.infolist():
                if info.is_dir() or not info.filename.endswith(".npy"):
                    raise ResultArchiveError(
                        f"{label} contains non-NPY member {info.filename!r}"
                    )
                if "/" in info.filename:
                    raise ResultArchiveError(
                        f"{label} contains nested array member {info.filename!r}"
                    )
                array_name = info.filename[:-4]
                if not array_name or array_name in array_names:
                    raise ResultArchiveError(
                        f"{label} contains duplicate or empty array names"
                    )
                array_names.add(array_name)
                with payload.open(info, "r") as member:
                    version = np.lib.format.read_magic(member)
                    if version == (1, 0):
                        shape, _fortran_order, dtype = (
                            np.lib.format.read_array_header_1_0(
                                member,
                                max_header_size=budget.maximum_npy_header_bytes,
                            )
                        )
                    elif version == (2, 0):
                        shape, _fortran_order, dtype = (
                            np.lib.format.read_array_header_2_0(
                                member,
                                max_header_size=budget.maximum_npy_header_bytes,
                            )
                        )
                    else:
                        raise ResultArchiveError(
                            f"{label} array {array_name!r} uses unsupported NPY version {version}"
                        )
                    header_and_prefix_bytes = member.tell()

                dtype = np.dtype(dtype)
                if (
                    dtype.fields is not None
                    or dtype.subdtype is not None
                    or dtype.kind not in {"b", "i", "u", "f", "c"}
                ):
                    raise ResultArchiveError(
                        f"{label} array {array_name!r} uses disallowed dtype {dtype}"
                    )
                if len(shape) > budget.maximum_array_dimensions:
                    raise ResultArchiveError(
                        f"{label} array {array_name!r} has too many dimensions"
                    )
                if any(
                    not isinstance(length, int)
                    or length < 0
                    or length > budget.maximum_axis_length
                    for length in shape
                ):
                    raise ResultArchiveError(
                        f"{label} array {array_name!r} has an invalid or oversized axis"
                    )
                element_count = math.prod(shape)
                array_bytes = element_count * dtype.itemsize
                if element_count > budget.maximum_array_elements:
                    raise ResultArchiveError(
                        f"{label} array {array_name!r} exceeds the element-count limit"
                    )
                if array_bytes > budget.maximum_array_bytes:
                    raise ResultArchiveError(
                        f"{label} array {array_name!r} exceeds the byte-size limit"
                    )
                if header_and_prefix_bytes + array_bytes != info.file_size:
                    raise ResultArchiveError(
                        f"{label} array {array_name!r} NPY header does not match its payload size"
                    )
                usage.array_count += 1
                usage.array_elements += element_count
                usage.array_bytes += array_bytes
                if usage.array_count > budget.maximum_array_count:
                    raise ResultArchiveError(
                        "result archive exceeds the total array-count limit"
                    )
                if usage.array_elements > budget.maximum_array_elements:
                    raise ResultArchiveError(
                        "result archive exceeds the total array-element limit"
                    )
                if usage.array_bytes > budget.maximum_total_array_bytes:
                    raise ResultArchiveError(
                        "result archive exceeds the total decoded-array byte limit"
                    )
    except ResultArchiveError:
        raise
    except (BadZipFile, EOFError, OSError, TypeError, UnicodeError, ValueError) as exc:
        raise ResultArchiveError(f"{label} is not a readable NPZ payload") from exc


def _copy_npz(
    data: bytes,
    budget: ArchiveResourceBudget,
    usage: _ArchiveReadUsage,
    *,
    label: str,
) -> dict[str, np.ndarray]:
    _inspect_npz(data, budget, usage, label=label)
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as payload:
            return {name: np.asarray(payload[name]).copy() for name in payload.files}
    except (BadZipFile, EOFError, OSError, TypeError, ValueError) as exc:
        raise ResultArchiveError(f"could not decode NumPy result arrays: {exc}") from exc


def _verify_payload_sha256(payload: bytes, expected: Any, *, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ResultArchiveError(
            f"{label} is missing a valid lowercase SHA-256 digest"
        )
    actual = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise ResultArchiveError(f"{label} failed the archived SHA-256 check")


def _read_grid(arrays: dict[str, np.ndarray], prefix: str) -> Grid3D:
    names = {
        axis: f"{prefix}{axis}_m" for axis in ("x", "y", "z")
    } | {
        f"{axis}_edges": f"{prefix}{axis}_edges_m" for axis in ("x", "y", "z")
    }
    missing = [name for name in names.values() if name not in arrays]
    if missing:
        raise ResultArchiveError(f"result grid is missing arrays: {missing}")
    x_edges = arrays[names["x_edges"]]
    y_edges = arrays[names["y_edges"]]
    z_edges = arrays[names["z_edges"]]
    if min(len(x_edges), len(y_edges), len(z_edges)) < 2:
        raise ResultArchiveError("result grid edge arrays must contain at least two values")
    try:
        return Grid3D(
            x=arrays[names["x"]],
            y=arrays[names["y"]],
            z=arrays[names["z"]],
            dx=float(np.mean(np.diff(x_edges))),
            dy=float(np.mean(np.diff(y_edges))),
            dz=float(np.mean(np.diff(z_edges))),
            x_edges=x_edges,
            y_edges=y_edges,
            z_edges=z_edges,
        )
    except (TypeError, ValueError) as exc:
        label = "physical" if not prefix else prefix.rstrip("_")
        raise ResultArchiveError(f"invalid {label} result grid: {exc}") from exc


def _read_result(
    archive: ZipFile,
    prefix: str,
    budget: ArchiveResourceBudget,
    usage: _ArchiveReadUsage,
) -> SimulationResult:
    from .solver import SimulationResult

    summary = _read_json(archive, f"{prefix}/summary.json", budget)
    metadata = _read_json(archive, f"{prefix}/result_metadata.json", budget)
    fields_name = f"{prefix}/fields.npz"
    if archive.getinfo(fields_name).file_size > budget.maximum_npz_bytes:
        raise ResultArchiveError(f"{fields_name} exceeds maximum_npz_bytes")
    fields_payload = archive.read(fields_name)
    _verify_payload_sha256(
        fields_payload,
        metadata.get("fields_sha256"),
        label="result fields",
    )
    arrays = _copy_npz(
        fields_payload,
        budget,
        usage,
        label=fields_name,
    )
    missing = [name for name in REQUIRED_RESULT_ARRAYS if name not in arrays]
    if missing:
        raise ResultArchiveError(f"completed result is missing required arrays: {missing}")
    declared_optional = metadata.get("optional_arrays", [])
    if not isinstance(declared_optional, list) or any(
        not isinstance(name, str) for name in declared_optional
    ):
        raise ResultArchiveError("result metadata optional_arrays must be a string list")
    if len(set(declared_optional)) != len(declared_optional):
        raise ResultArchiveError("result metadata optional_arrays contains duplicates")
    unknown_optional = set(declared_optional) - set(OPTIONAL_RESULT_ARRAYS)
    if unknown_optional:
        raise ResultArchiveError(
            f"result metadata declares unknown optional arrays: {sorted(unknown_optional)}"
        )
    present_optional = {
        name for name in OPTIONAL_RESULT_ARRAYS if name in arrays
    }
    if set(declared_optional) != present_optional:
        raise ResultArchiveError(
            "result metadata optional_arrays does not match the saved field payload"
        )

    grid = _read_grid(arrays, "")
    has_optical_grid = metadata.get("has_optical_grid", False)
    has_pump_optical_grid = metadata.get("has_pump_optical_grid", False)
    if not isinstance(has_optical_grid, bool) or not isinstance(
        has_pump_optical_grid, bool
    ):
        raise ResultArchiveError(
            "result metadata optical-grid availability flags must be boolean"
        )
    optical_grid = (
        _read_grid(arrays, "optical_")
        if has_optical_grid
        else None
    )
    pump_optical_grid = (
        _read_grid(arrays, "pump_optical_")
        if has_pump_optical_grid
        else None
    )
    raw_level_names = metadata.get("level_names", [])
    if not isinstance(raw_level_names, list):
        raise ResultArchiveError("result metadata level_names must be a list")
    level_names = [str(name) for name in raw_level_names]
    if not level_names or any(not name for name in level_names):
        raise ResultArchiveError("completed result must define non-empty level_names")
    if len(set(level_names)) != len(level_names):
        raise ResultArchiveError("completed result level_names must be unique")

    stress_entries = metadata.get("stress_arrays", [])
    if not isinstance(stress_entries, list):
        raise ResultArchiveError("result metadata stress_arrays must be a list")
    stress: dict[str, np.ndarray] = {}
    for item in stress_entries:
        if not isinstance(item, dict) or "name" not in item or "key" not in item:
            raise ResultArchiveError("each stress array entry must define name and key")
        name = str(item["name"])
        if name in stress:
            raise ResultArchiveError(f"duplicate stress array name {name!r}")
        key = str(item["key"])
        if key not in arrays:
            raise ResultArchiveError(f"stress array {name!r} is missing payload {key!r}")
        stress[name] = arrays[key]
    optical_planes: dict[str, np.ndarray] = {}
    optical_plane_coordinates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    plane_entries = metadata.get("optical_planes", [])
    if not isinstance(plane_entries, list):
        raise ResultArchiveError("result metadata optical_planes must be a list")
    if plane_entries:
        plane_name = f"{prefix}/optical_planes.npz"
        if archive.getinfo(plane_name).file_size > budget.maximum_npz_bytes:
            raise ResultArchiveError(f"{plane_name} exceeds maximum_npz_bytes")
        plane_payload = archive.read(plane_name)
        _verify_payload_sha256(
            plane_payload,
            metadata.get("optical_planes_sha256"),
            label="result optical planes",
        )
        plane_arrays = _copy_npz(
            plane_payload,
            budget,
            usage,
            label=plane_name,
        )
        for item in plane_entries:
            if not isinstance(item, dict) or "name" not in item or "key" not in item:
                raise ResultArchiveError(
                    "each optical plane entry must define name and key"
                )
            name = str(item["name"])
            if name in optical_planes:
                raise ResultArchiveError(f"duplicate optical plane name {name!r}")
            plane_key = str(item["key"])
            if plane_key not in plane_arrays:
                raise ResultArchiveError(
                    f"optical plane {name!r} is missing payload {plane_key!r}"
                )
            optical_planes[name] = plane_arrays[plane_key]
            coordinate_keys = item.get("coordinate_keys")
            if coordinate_keys is not None:
                if not isinstance(coordinate_keys, list) or len(coordinate_keys) != 2:
                    raise ResultArchiveError(
                        f"optical plane {name!r} coordinate_keys must contain x and y"
                    )
                missing_coordinates = [
                    str(key) for key in coordinate_keys if str(key) not in arrays
                ]
                if missing_coordinates:
                    raise ResultArchiveError(
                        f"optical plane {name!r} is missing coordinate arrays "
                        f"{missing_coordinates}"
                    )
                optical_plane_coordinates[name] = (
                    arrays[str(coordinate_keys[0])],
                    arrays[str(coordinate_keys[1])],
                )

    _validate_result_array_shapes(
        arrays,
        grid,
        stress,
        optical_planes,
        optical_plane_coordinates,
        optical_grid,
        pump_optical_grid,
        level_names,
    )
    _validate_intensity_snapshots(arrays, summary)

    required = {name: arrays[name] for name in REQUIRED_RESULT_ARRAYS}
    optional = {
        name: arrays.get(name)
        for name in OPTIONAL_RESULT_ARRAYS
    }
    return SimulationResult(
        grid=grid,
        stress=stress,
        optical_planes=optical_planes,
        optical_plane_coordinates=optical_plane_coordinates,
        level_names=level_names,
        summary=summary,
        cpa_diagnostics=metadata.get("cpa_diagnostics"),
        optical_grid=optical_grid,
        pump_optical_grid=pump_optical_grid,
        **required,
        **optional,
    )


def _validate_intensity_snapshots(
    arrays: dict[str, np.ndarray], summary: dict[str, Any],
) -> None:
    """Keep native intensity planes inseparable from their coordinates/times."""
    names = (
        "longitudinal_intensity_W_m2", "longitudinal_intensity_area_m2",
        "longitudinal_temporal_power_W",
    )
    transport = (summary.get("signal") or {}).get("gain_transport") or {}
    metadata = transport.get("intensity_snapshots", {})
    if not any(name in arrays for name in names) and not metadata:
        return  # Legacy archives and runs with recording disabled.
    if not all(name in arrays for name in names) or not isinstance(metadata, dict):
        raise ResultArchiveError("intensity snapshots require all arrays and metadata")
    intensity, area, power = (np.asarray(arrays[name]) for name in names)
    if intensity.ndim != 4 or intensity.shape[1] != 2 or min(intensity.shape) < 1:
        raise ResultArchiveError("intensity snapshots must have shape (plane, 2, a, b)")
    if area.shape != intensity.shape[-2:]:
        raise ResultArchiveError("intensity snapshot area must match the native transverse grid")
    for values in (intensity, area, power):
        if np.iscomplexobj(values) or np.any(values < 0.0):
            raise ResultArchiveError("intensity snapshot arrays must be real and nonnegative")
    if not np.any(area > 0.0):
        raise ResultArchiveError("intensity snapshots require positive integration area")
    try:
        times = np.asarray(metadata["time_s"], dtype=float)
        planes = metadata["planes"]
        grids = metadata["grids_by_pass"]
        if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)):
            raise ValueError("invalid time axis")
        if not np.all(np.diff(times) > 0.0):
            raise ValueError("time axis must increase")
        if power.shape != (intensity.shape[0], times.size):
            raise ValueError("temporal power shape disagrees with time/plane axes")
        if len(planes) != intensity.shape[0]:
            raise ValueError("plane metadata count disagrees with intensity array")
        for index, plane in enumerate(planes):
            if _require_json_integer(plane["plane_index"], label="plane_index") != index:
                raise ValueError("plane indices must follow array order")
            pass_number = _require_json_integer(plane["pass"], label="intensity pass", minimum=1)
            geometry = grids[str(pass_number)]
            coordinate_names = (
                ("theta_rad", "radius_m") if geometry["coordinate_system"] == "polar"
                else ("y_m", "x_m") if geometry["coordinate_system"] == "cartesian"
                else ()
            )
            if not coordinate_names:
                raise ValueError("unknown native coordinate system")
            for name, count in zip(coordinate_names, intensity.shape[-2:]):
                coordinate = np.asarray(geometry[name], dtype=float)
                if coordinate.shape != (count,) or not np.all(np.isfinite(coordinate)):
                    raise ValueError("native coordinates disagree with intensity array")
            cosine = float(plane["projected_area_cosine"])
            if not 0.0 < cosine <= 1.0:
                raise ValueError("invalid projected area cosine")
            if len(plane["selections"]) != 2:
                raise ValueError("two time selections are required")
            for selection in plane["selections"]:
                time_index = _require_json_integer(selection["time_index"], label="time_index")
                if time_index >= times.size or float(selection["time_s"]) != times[time_index]:
                    raise ValueError("selected time disagrees with time axis")
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
        raise ResultArchiveError(f"invalid intensity snapshot metadata: {exc}") from exc


def _validate_result_array_shapes(
    arrays: dict[str, np.ndarray],
    grid: Grid3D,
    stress: dict[str, np.ndarray],
    optical_planes: dict[str, np.ndarray],
    optical_plane_coordinates: dict[str, tuple[np.ndarray, np.ndarray]],
    optical_grid: Grid3D | None,
    pump_optical_grid: Grid3D | None,
    level_names: list[str],
) -> None:
    """Reject archives whose fields no longer agree with their grids."""

    for name in (*REQUIRED_RESULT_ARRAYS, *OPTIONAL_RESULT_ARRAYS):
        if name not in arrays:
            continue
        try:
            finite = np.isfinite(np.asarray(arrays[name]))
        except TypeError as exc:
            raise ResultArchiveError(
                f"result array {name} must have a numeric dtype"
            ) from exc
        if not np.all(finite):
            raise ResultArchiveError(
                f"result array {name} must contain only finite values"
            )

    for name in REQUIRED_RESULT_ARRAYS:
        values = np.asarray(arrays[name])
        expected_ndim = 4 if name in {
            "population_m3",
            "population_after_pump_m3",
        } else 3
        if values.ndim != expected_ndim or values.shape[-3:] != grid.shape:
            raise ResultArchiveError(
                f"result array {name} must have {expected_ndim} dimensions and "
                f"end in grid shape {grid.shape}, "
                f"got {values.shape}"
            )
    for name in ("population_m3", "population_after_pump_m3"):
        if np.asarray(arrays[name]).shape[0] != len(level_names):
            raise ResultArchiveError(
                f"result array {name} leading dimension must match "
                f"{len(level_names)} level_names"
            )

    volume_optional = {
        "crystal_mask",
        "pump_active_absorbed_W_m3",
        "pump_passive_absorbed_W_m3",
        "signal_fluence_last_pass_J_m2",
        "signal_fluence_cumulative_J_m2",
        "dopant_density_m3",
        "nonlinear_loss_energy_J_m3",
        "ase_intensity_W_m2",
        "signal_passive_loss_energy_J_m3",
    }
    stacked_volume_optional = {
        "signal_fluence_per_pass_J_m2",
        "pump_spectral_intensity_W_m2",
        "pump_spectral_active_absorbed_W_m3",
        "pump_spectral_passive_absorbed_W_m3",
    }
    for name in volume_optional:
        if name not in arrays:
            continue
        values = np.asarray(arrays[name])
        if values.ndim != 3 or values.shape != grid.shape:
            raise ResultArchiveError(
                f"result array {name} must equal grid shape {grid.shape}, "
                f"got {values.shape}"
            )
    for name in stacked_volume_optional:
        if name not in arrays:
            continue
        values = np.asarray(arrays[name])
        if values.ndim != 4 or values.shape[-3:] != grid.shape:
            raise ResultArchiveError(
                f"result array {name} must have one leading channel/pass axis "
                f"followed by grid shape {grid.shape}, got {values.shape}"
            )

    if "output_jones_field" in arrays:
        shape = np.asarray(arrays["output_jones_field"]).shape
        allowed_transverse_shapes = {grid.shape[-2:]}
        if optical_grid is not None:
            allowed_transverse_shapes.add(
                (len(optical_grid.y), len(optical_grid.x))
            )
        if len(shape) < 2 or shape[-2:] not in allowed_transverse_shapes:
            raise ResultArchiveError(
                "result array output_jones_field must end in a saved Seed/physical "
                f"grid (ny, nx) shape; got {shape[-2:]}, expected one of "
                f"{sorted(allowed_transverse_shapes)}"
            )

    one_dimensional = {
        "wavelengths_m",
        "input_spectrum_J",
        "output_spectrum_J",
        "time_s",
        "input_temporal_energy_J",
        "output_temporal_energy_J",
        "frequency_Hz",
        "input_spectral_phase_rad",
        "output_spectral_phase_rad",
        "compressed_spectral_phase_rad",
        "compressed_temporal_power_W",
        "pump_wavelengths_m",
    }
    for name in one_dimensional:
        if name in arrays and np.asarray(arrays[name]).ndim != 1:
            raise ResultArchiveError(f"result array {name} must be one-dimensional")
    for name in ("wavelengths_m", "time_s", "frequency_Hz", "pump_wavelengths_m"):
        if name in arrays and not np.all(np.isfinite(arrays[name])):
            raise ResultArchiveError(
                f"result coordinate array {name} must contain only finite values"
            )
    for coordinate, dependents in (
        ("wavelengths_m", ("input_spectrum_J", "output_spectrum_J")),
        (
            "time_s",
            (
                "input_temporal_energy_J",
                "output_temporal_energy_J",
                "compressed_temporal_power_W",
            ),
        ),
        (
            "frequency_Hz",
            (
                "input_spectral_phase_rad",
                "output_spectral_phase_rad",
                "compressed_spectral_phase_rad",
            ),
        ),
    ):
        for name in dependents:
            if name in arrays and coordinate not in arrays:
                raise ResultArchiveError(
                    f"result array {name} requires coordinate array {coordinate}"
                )
        if coordinate not in arrays:
            continue
        count = len(arrays[coordinate])
        for name in dependents:
            if name in arrays and len(arrays[name]) != count:
                raise ResultArchiveError(
                    f"result array {name} length does not match {coordinate}"
                )

    for edge_name, coordinate in (
        ("temporal_bin_edges_s", "time_s"),
        ("spectral_bin_edges_Hz", "wavelengths_m"),
    ):
        if edge_name not in arrays:
            continue
        edges = np.asarray(arrays[edge_name])
        if coordinate not in arrays or edges.shape != (len(arrays[coordinate]) + 1,):
            raise ResultArchiveError(f"{edge_name} must contain one more edge than {coordinate}")
        if np.any(np.diff(edges) <= 0.0) or (
            edge_name == "spectral_bin_edges_Hz" and np.any(edges <= 0.0)
        ):
            raise ResultArchiveError(f"{edge_name} must contain strictly increasing valid edges")

    pump_spectral_arrays = (
        "pump_spectral_intensity_W_m2",
        "pump_spectral_active_absorbed_W_m3",
        "pump_spectral_passive_absorbed_W_m3",
    )
    for name in pump_spectral_arrays:
        if name in arrays and "pump_wavelengths_m" not in arrays:
            raise ResultArchiveError(
                f"result array {name} requires pump_wavelengths_m"
            )
    if "pump_wavelengths_m" in arrays:
        count = len(arrays["pump_wavelengths_m"])
        for name in pump_spectral_arrays:
            if name in arrays and np.asarray(arrays[name]).shape[0] != count:
                raise ResultArchiveError(
                    f"result array {name} leading dimension does not match pump_wavelengths_m"
                )

    for name, values in stress.items():
        stress_values = np.asarray(values)
        shape = stress_values.shape
        try:
            has_infinity = bool(np.any(np.isinf(stress_values)))
        except TypeError as exc:
            raise ResultArchiveError(
                f"stress array {name} must have a numeric dtype"
            ) from exc
        if has_infinity:
            raise ResultArchiveError(
                f"stress array {name} must not contain infinite values"
            )
        if len(shape) >= 3 and shape[-3:] != grid.shape:
            raise ResultArchiveError(
                f"stress array {name} must end in grid shape {grid.shape}, got {shape}"
            )
        if len(shape) == 2 and (
            "surface" in name or "endface" in name
        ) and shape != grid.shape[-2:]:
            raise ResultArchiveError(
                f"stress surface array {name} must match grid (ny, nx) {grid.shape[-2:]}"
            )

    for name, values in optical_planes.items():
        plane_values = np.asarray(values)
        shape = plane_values.shape
        try:
            plane_is_finite = bool(np.all(np.isfinite(plane_values)))
        except TypeError as exc:
            raise ResultArchiveError(
                f"optical plane {name} must have a numeric dtype"
            ) from exc
        if not plane_is_finite:
            raise ResultArchiveError(
                f"optical plane {name} must contain only finite values"
            )
        if len(shape) != 2:
            raise ResultArchiveError(
                f"optical plane {name} must be two-dimensional, got {shape}"
            )
        if name in optical_plane_coordinates:
            x, y = optical_plane_coordinates[name]
            if np.asarray(x).ndim != 1 or np.asarray(y).ndim != 1:
                raise ResultArchiveError(
                    f"optical plane {name} coordinates must be one-dimensional"
                )
            if len(x) == 0 or len(y) == 0:
                raise ResultArchiveError(
                    f"optical plane {name} coordinates must not be empty"
                )
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                raise ResultArchiveError(
                    f"optical plane {name} coordinates must be finite"
                )
            expected = (len(y), len(x))
            if shape != expected:
                raise ResultArchiveError(
                    f"optical plane {name} must match coordinate shape {expected}, got {shape}"
                )
            continue
        candidate_grids = (
            (pump_optical_grid, optical_grid, grid)
            if str(name).lower().startswith("pump")
            else (optical_grid, grid, pump_optical_grid)
        )
        candidate_shapes = {
            (len(candidate.y), len(candidate.x))
            for candidate in candidate_grids
            if candidate is not None
        }
        if shape not in candidate_shapes:
            raise ResultArchiveError(
                f"optical plane {name} shape {shape} matches neither explicit "
                f"coordinates nor an archived grid {sorted(candidate_shapes)}"
            )
