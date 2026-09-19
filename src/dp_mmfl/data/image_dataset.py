from io import BytesIO
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from dp_mmfl.data.labels import TARGET_COLUMNS


class CheXpertPlusImageDataset(Dataset):
    """
    Image-only PyTorch Dataset for CheXpert Plus frontal chest radiographs.
    Streams images on-demand from Redivis and provides 13 pathology targets with U-Ignore masks.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        table,
        split: str,
        image_transform=None,
    ):
        manifest = pd.read_parquet(manifest_path)

        manifest = manifest[
            (manifest["experiment_split"] == split)
            & (manifest["frontal_lateral"] == "Frontal")
        ].reset_index(drop=True)

        self.manifest = manifest
        self.table = table
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_image(self, path: str) -> Image.Image:
        png_path = str(path)

        if png_path.startswith("train/"):
            png_path = png_path[len("train/"):]

        if png_path.endswith(".jpg"):
            png_path = png_path[:-4] + ".png"

        file = self.table.file(png_path)
        data = file.read(as_text=False)

        image = Image.open(BytesIO(data)).convert("L")
        return image

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]

        image = self._load_image(row["path_to_image"])

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
