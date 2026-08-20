"""Utilities for processing PROCS simulation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, sosfilt_zi

DEFAULT_TIME_COLUMN = 't_s'
DEFAULT_FORCE_COLUMN = 'contact_force_N'
DEFAULT_CUTOFF_HZ = 20.0
DEFAULT_ORDER = 6


def filter_contact_force(
    time_s: np.ndarray,
    force_n: np.ndarray,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    order: int = DEFAULT_ORDER,
) -> np.ndarray:
    """Apply a single-pass Butterworth filter to a complete offline trajectory."""
    time_s = np.asarray(time_s, dtype=float)
    force_n = np.asarray(force_n, dtype=float)
    if time_s.ndim != 1 or force_n.ndim != 1 or time_s.size != force_n.size:
        raise ValueError('time and force must be one-dimensional arrays of equal length')
    if time_s.size < 2:
        raise ValueError('at least two samples are required to infer the sample rate')
    if not np.isfinite(time_s).all() or not np.isfinite(force_n).all():
        raise ValueError('time and force must contain only finite values')

    dt = np.diff(time_s)
    if (dt <= 0).any() or not np.allclose(dt, dt[0], rtol=1e-6, atol=max(abs(dt[0]) * 1e-9, 1e-12)):
        raise ValueError('time samples must be strictly increasing and uniformly spaced')
    if not np.isfinite(cutoff_hz) or cutoff_hz <= 0:
        raise ValueError(f'cutoff_hz must be positive and finite, got {cutoff_hz}')
    if isinstance(order, bool) or int(order) != order or order <= 0:
        raise ValueError(f'order must be a positive integer, got {order}')

    fs = 1.0 / dt[0]
    if cutoff_hz >= fs / 2.0:
        raise ValueError(f'cutoff_hz must be below Nyquist ({fs / 2.0:g} Hz), got {cutoff_hz}')

    sos = butter(int(order), cutoff_hz, fs=fs, output='sos')
    zi = sosfilt_zi(sos) * force_n[0]
    return sosfilt(sos, force_n, zi=zi)[0]


def contact_force_metrics(time_s: np.ndarray, force_n: np.ndarray) -> dict[str, float]:
    """Return raw, 20 Hz filtered, and contact-loss trajectory metrics."""
    time_s = np.asarray(time_s, dtype=float)
    force_n = np.asarray(force_n, dtype=float)
    filtered = filter_contact_force(time_s, force_n)
    dt = float(time_s[1] - time_s[0])
    longest_loss = current_loss = 0
    for offline in force_n <= 0.0:
        current_loss = current_loss + 1 if offline else 0
        longest_loss = max(longest_loss, current_loss)
    return {
        'raw_mean_N': float(force_n.mean()),
        'raw_std_N': float(force_n.std()),
        'raw_min_N': float(force_n.min()),
        'raw_max_N': float(force_n.max()),
        'filtered_mean_N': float(filtered.mean()),
        'filtered_std_N': float(filtered.std()),
        'filtered_min_N': float(filtered.min()),
        'filtered_max_N': float(filtered.max()),
        'loss_fraction': float((force_n <= 0.0).mean()),
        'longest_loss_s': longest_loss * dt,
    }


def filter_csv(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    time_column: str = DEFAULT_TIME_COLUMN,
    force_column: str = DEFAULT_FORCE_COLUMN,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    order: int = DEFAULT_ORDER,
) -> Path:
    """Read a raw trajectory CSV and write it with one filtered-force column."""
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(f'{input_path.stem}_filtered{input_path.suffix}')
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f'output already exists: {output_path}')

    frame = pd.read_csv(input_path)
    missing = [column for column in (time_column, force_column) if column not in frame.columns]
    if missing:
        raise ValueError(f'missing required CSV column(s): {", ".join(missing)}')
    filtered_column = f'{force_column}_filtered'
    frame[filtered_column] = filter_contact_force(
        frame[time_column].to_numpy(),
        frame[force_column].to_numpy(),
        cutoff_hz=cutoff_hz,
        order=order,
    )
    frame.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_csv', type=Path)
    parser.add_argument('-o', '--output', type=Path)
    parser.add_argument('--time-column', default=DEFAULT_TIME_COLUMN)
    parser.add_argument('--force-column', default=DEFAULT_FORCE_COLUMN)
    parser.add_argument('--cutoff-hz', type=float, default=DEFAULT_CUTOFF_HZ)
    parser.add_argument('--order', type=int, default=DEFAULT_ORDER)
    args = parser.parse_args()
    output = filter_csv(
        args.input_csv,
        args.output,
        time_column=args.time_column,
        force_column=args.force_column,
        cutoff_hz=args.cutoff_hz,
        order=args.order,
    )
    print(f'Filtered CSV written -> {output}')


if __name__ == '__main__':
    main()
