from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CachedAudioDataset(Dataset):
    """Read cached features using original row positions in the full index."""

    def __init__(
        self,
        window_index: pd.DataFrame,
        cache_path,
        full_window_index: pd.DataFrame,
        target_shape=(2, 128, 257),
    ):
        required = {"window_id", "sample_id", "label", "split"}
        for name, frame in (
            ("window_index", window_index),
            ("full_window_index", full_window_index),
        ):
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"{name} is missing columns: {sorted(missing)}")

        self.window_index = window_index.reset_index(drop=True).copy()
        self.cache_path = Path(cache_path)
        self.target_shape = tuple(target_shape)

        if not self.cache_path.is_file():
            raise FileNotFoundError(f"Feature cache not found: {self.cache_path}")

        full_index = full_window_index.reset_index(drop=True).copy()

        # Window IDs must uniquely identify cache rows.
        if full_index["window_id"].duplicated().any():
            raise ValueError("Full window index contains duplicate window_id values.")

        full_positions = pd.Series(
            np.arange(len(full_index), dtype=np.int64),
            index=full_index["window_id"].astype(str),
        )

        subset_ids = self.window_index["window_id"].astype(str)
        if subset_ids.duplicated().any():
            raise ValueError("Subset contains duplicate window_id values.")

        mapped_positions = subset_ids.map(full_positions)
        if mapped_positions.isna().any():
            missing_ids = subset_ids[mapped_positions.isna()].head(5).tolist()
            raise ValueError(
                f"Subset window IDs are absent from full index; examples: {missing_ids}"
            )

        self.cache_positions = mapped_positions.to_numpy(dtype=np.int64)

        # Verify each subset row's split and label against the full index.
        full_by_id = full_index.set_index(
            full_index["window_id"].astype(str), drop=False
        )
        for column in ("label", "split", "sample_id"):
            expected = subset_ids.map(full_by_id[column].astype(str)).to_numpy()
            actual = self.window_index[column].astype(str).to_numpy()
            if not np.array_equal(actual, expected):
                raise ValueError(
                    f"Subset {column!r} values disagree with full index."
                )

        expected_bytes = len(full_index) * int(np.prod(self.target_shape)) * 2
        actual_bytes = self.cache_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Cache size mismatch: expected {expected_bytes} bytes, "
                f"found {actual_bytes} bytes."
            )

        self.feature_cache = np.memmap(
            self.cache_path,
            dtype=np.float16,
            mode="r",
            shape=(len(full_index), *self.target_shape),
        )

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
        row = self.window_index.iloc[idx]
        cache_position = int(self.cache_positions[idx])

        features = np.asarray(
            self.feature_cache[cache_position], dtype=np.float32
        ).copy()

        if features.shape != self.target_shape:
            raise ValueError(
                f"Unexpected feature shape at cache row {cache_position}: "
                f"{features.shape}"
            )
        if not np.isfinite(features).all():
            raise ValueError(
                f"Non-finite features at cache row {cache_position}"
            )

        label_text = str(row["label"]).strip().lower()
        if label_text == "bonafide":
            label = 0
        elif label_text == "spoof":
            label = 1
        else:
            raise ValueError(
                f"Unsupported label {row['label']!r} for window {row['window_id']}"
            )

        return {
            "features": torch.from_numpy(features),
            "label": torch.tensor(label, dtype=torch.long),
            "window_id": row["window_id"],
            "sample_id": row["sample_id"],
            "cache_position": cache_position,
        }
