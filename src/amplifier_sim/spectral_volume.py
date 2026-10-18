"""Bounded spectral-volume reads for reduced spatial transport.

No implicit ndarray conversion: callers must consume blocks rather than
accidentally expanding the full wavelength × z × y × x volume.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
import numpy as np


class SpectralIntensityVolume:
    def __init__(self, shape: tuple[int, ...], blocks: Callable[[], Iterator[tuple[slice, np.ndarray]]], *, stored_nbytes: int):
        self.shape = tuple(shape)
        self._blocks = blocks
        self.stored_nbytes = int(stored_nbytes)

    def iter_blocks(self):
        yield from self._blocks()

    @classmethod
    def reduced(cls, values: np.ndarray, spatial_shape: tuple[int, ...], expand: Callable, *, block_bytes: int = 32 * 1024**2):
        per_channel = int(np.prod(spatial_shape)) * values.dtype.itemsize
        block_size = max(1, block_bytes // max(per_channel, 1))
        def blocks():
            for start in range(0, len(values), block_size):
                selection = slice(start, min(start + block_size, len(values)))
                yield selection, expand(values[selection])
        return cls((len(values), *spatial_shape), blocks, stored_nbytes=values.nbytes)

    def mapped(self, mapper: Callable, spatial_shape: tuple[int, ...]):
        per_channel = int(np.prod(spatial_shape)) * np.dtype(float).itemsize
        block_size = max(1, (32 * 1024**2) // max(per_channel, 1))
        def blocks():
            for selection, values in self.iter_blocks():
                for start in range(0, len(values), block_size):
                    stop = min(start + block_size, len(values))
                    yield slice(selection.start + start, selection.start + stop), mapper(values[start:stop])
        return type(self)((self.shape[0], *spatial_shape), blocks, stored_nbytes=self.stored_nbytes)


def spectral_intensity_blocks(values, wavelengths, spatial_shape):
    """Validate both representations; keep the dense numerical path unchanged."""
    expected = (len(wavelengths), *spatial_shape)
    if isinstance(values, SpectralIntensityVolume):
        if values.shape != expected:
            raise ValueError(f"signal_spectral_intensity_W_m2 must have shape {expected}")
        end = 0
        for selection, block in values.iter_blocks():
            if selection.start != end or selection.step not in (None, 1):
                raise ValueError("spectral blocks must cover wavelengths in order")
            end = selection.stop
            if end > len(wavelengths) or block.shape != (end - selection.start, *spatial_shape):
                raise ValueError("spectral intensity block has an invalid shape")
            yield wavelengths[selection], block
        if end != len(wavelengths):
            raise ValueError("spectral blocks do not cover all wavelengths")
    else:
        array = np.asarray(values, dtype=float)
        if array.shape != expected:
            raise ValueError(f"signal_spectral_intensity_W_m2 must have shape {expected}")
        yield wavelengths, array


def concatenate_spectral_intensities(volumes: Sequence):
    if not any(isinstance(value, SpectralIntensityVolume) for value in volumes):
        return np.concatenate(volumes, axis=0)
    spatial_shape = volumes[0].shape[1:]
    if any(value.shape[1:] != spatial_shape for value in volumes):
        raise ValueError("spectral intensity volumes must share the spatial grid")
    def blocks():
        offset = 0
        for value in volumes:
            if isinstance(value, SpectralIntensityVolume):
                for selection, block in value.iter_blocks():
                    yield slice(offset + selection.start, offset + selection.stop), block
            else:
                # ASE may already be dense; do not copy or concatenate it.
                for index in range(len(value)):
                    yield slice(offset + index, offset + index + 1), value[index:index + 1]
            offset += value.shape[0]
    return SpectralIntensityVolume(
        (sum(value.shape[0] for value in volumes), *spatial_shape), blocks,
        stored_nbytes=sum(value.stored_nbytes if isinstance(value, SpectralIntensityVolume) else value.nbytes for value in volumes),
    )
