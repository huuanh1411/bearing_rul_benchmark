from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def convert_log_directory(
    input_dir: Path,
    output_path: Path,
    pattern: str = "*.csv",
    has_header: bool = False,
    feature_columns: list[str] | None = None,
    max_files: int | None = None,
    row_step: int = 1,
    chunk_size: int = 250_000,
    progress_callback: callable | None = None,
) -> Path:
    if row_step < 1:
        raise ValueError("row_step must be at least 1.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")

    files = sorted(input_dir.glob(pattern))
    if max_files is not None:
        files = files[:max_files]

    if not files:
        raise ValueError(f"No files matched pattern {pattern!r} in {input_dir}")

    expected_column_count: int | None = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    write_header = True

    for file_index, file_path in enumerate(files, start=1):
        total_rows = _count_rows(file_path, has_header)
        row_offset = 0
        sampled_rows = 0

        if progress_callback is not None:
            progress_callback(f"[{file_index}/{len(files)}] Processing {file_path.name} ({total_rows} rows)")

        for chunk in pd.read_csv(file_path, header=0 if has_header else None, chunksize=chunk_size):
            if expected_column_count is None:
                expected_column_count = chunk.shape[1]
            elif chunk.shape[1] != expected_column_count:
                raise ValueError(
                    f"File {file_path.name} has {chunk.shape[1]} columns, expected {expected_column_count}."
                )

            chunk_positions = row_offset + np.arange(len(chunk), dtype=np.int64)
            if row_step > 1:
                selected = (chunk_positions % row_step) == 0
                chunk = chunk.iloc[selected].copy()
                chunk_positions = chunk_positions[selected]
            else:
                chunk = chunk.copy()

            row_offset += len(chunk_positions) if row_step == 1 else len(selected)

            if chunk.empty:
                continue

            sampled_rows += len(chunk)

            resolved_feature_columns = _resolve_feature_columns(
                chunk.shape[1],
                feature_columns,
                has_header,
                chunk.columns,
            )
            chunk.columns = resolved_feature_columns

            prepared = chunk
            prepared.insert(0, "rul", total_rows - 1 - chunk_positions)
            prepared.insert(0, "time_idx", chunk_positions)
            prepared.insert(0, "bearing_id", file_path.stem)
            prepared.to_csv(output_path, mode="a", header=write_header, index=False)
            write_header = False

        if progress_callback is not None:
            progress_callback(f"[{file_index}/{len(files)}] Finished {file_path.name} -> kept {sampled_rows} rows")

    return output_path


def _count_rows(file_path: Path, has_header: bool) -> int:
    with file_path.open("rb") as handle:
        row_count = sum(1 for _ in handle)
    if has_header:
        row_count -= 1
    return max(row_count, 0)


def _resolve_feature_columns(
    column_count: int,
    feature_columns: list[str] | None,
    has_header: bool,
    original_columns: pd.Index,
) -> list[str]:
    if feature_columns is not None:
        if len(feature_columns) != column_count:
            raise ValueError(
                f"Received {len(feature_columns)} feature column names, expected {column_count}."
            )
        return feature_columns

    if has_header:
        return [str(column_name) for column_name in original_columns]

    return [f"feature_{index}" for index in range(column_count)]