from pathlib import Path
from typing import Sequence, Union
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from dp_mmfl.data.labels import TARGET_COLUMNS


class CachedCheXpertPlusImageDataset(Dataset):
    """
    Cached image-only PyTorch Dataset for CheXpert Plus frontal radiographs.
    Loads locally stored PNG files indexed by 7-digit zero-padded sample_id.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        cache_root: Union[str, Path],
        sample_ids: Sequence[int],
        image_transform=None,
    ):
        manifest = pd.read_parquet(manifest_path)

        manifest = manifest[
            manifest["sample_id"].isin(sample_ids)
        ].copy()

        # Preserve the exact requested sample order
        order = {
            int(sample_id): i
            for i, sample_id in enumerate(sample_ids)
        }

        manifest["_order"] = manifest["sample_id"].map(order)
        manifest = (
            manifest
            .sort_values("_order")
            .drop(columns="_order")
            .reset_index(drop=True)
        )

        self.manifest = manifest
        self.cache_root = Path(cache_root)
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]

        image_path = (
            self.cache_root
            / f"{int(row['sample_id']):07d}.png"
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Cached image not found: {image_path}"
            )

        image = Image.open(image_path).convert("L")

        if self.image_transform is not None:
            image = self.image_transform(image)

        labels = torch.tensor(
            [
                0.0
                if pd.isna(row[f"target_{label}"])
                else float(row[f"target_{label}"])
                for label in TARGET_COLUMNS
            ],
            dtype=torch.float32,
        )

        label_mask = torch.tensor(
            [
                float(row[f"mask_{label}"])
                for label in TARGET_COLUMNS
            ],
            dtype=torch.float32,
        )

        return {
            "image": image,
            "labels": labels,
            "label_mask": label_mask,
            "sample_id": row["sample_id"],
            "patient_id": row["deid_patient_id"],
        }
