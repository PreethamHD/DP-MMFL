from io import BytesIO
from pathlib import Path
from typing import Optional, Union
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from dp_mmfl.data.labels import TARGET_COLUMNS
from dp_mmfl.data.text import ClinicalTextTokenizer


class CheXpertPlusMultimodalDataset(Dataset):
    """
    Multimodal PyTorch Dataset for CheXpert Plus.
    Streams frontal chest X-rays on-demand from Redivis and tokenizes clinical reports.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        table,
        image_transform=None,
        tokenizer: Optional[ClinicalTextTokenizer] = None,
        split: Optional[str] = None,
    ):
        self.manifest = pd.read_parquet(manifest_path)
        
        # Filter for frontal projections only
        self.manifest = self.manifest[
            self.manifest["frontal_lateral"] == "Frontal"
        ].copy()

        # Optional filter by experiment split
        if split is not None:
            self.manifest = self.manifest[
                self.manifest["experiment_split"] == split
            ].copy()

        self.manifest = self.manifest.reset_index(drop=True)
        self.table = table
        self.image_transform = image_transform
        self.tokenizer = tokenizer or ClinicalTextTokenizer()

    def __len__(self) -> int:
        return len(self.manifest)

    def _map_to_redivis_path(self, path_to_image: str) -> str:
        png_path = str(path_to_image)
        if png_path.startswith("train/"):
            png_path = png_path[len("train/"):]
        if png_path.endswith(".jpg"):
            png_path = png_path[:-4] + ".png"
        return png_path

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]

        # 1. Fetch and process image
        png_path = self._map_to_redivis_path(row["path_to_image"])
        file = self.table.file(png_path)
        data = file.read(as_text=False)
        image = Image.open(BytesIO(data)).convert("L")

        if self.image_transform is not None:
            image_tensor = self.image_transform(image)
        else:
            image_tensor = image

        # 2. Tokenize report_clean
        text_encoded = self.tokenizer.encode(str(row["report_clean"]))
        input_ids = text_encoded["input_ids"].squeeze(0)
        attention_mask = text_encoded["attention_mask"].squeeze(0)

        # 3. Extract 13 multi-label targets and U-Ignore masks
        target_cols = [f"target_{col}" for col in TARGET_COLUMNS]
        mask_cols = [f"mask_{col}" for col in TARGET_COLUMNS]

        labels = torch.tensor(row[target_cols].to_numpy(dtype=float), dtype=torch.float32)
        label_mask = torch.tensor(row[mask_cols].to_numpy(dtype=float), dtype=torch.float32)

        # 4. Extract patient metadata
        metadata = {
            "sample_id": row["sample_id"],
            "patient_id": row["deid_patient_id"],
            "age": row["age"],
            "sex": row["sex"],
            "race": row["race"],
            "ethnicity": row["ethnicity"],
            "experiment_split": row["experiment_split"],
            "frontal_lateral": row["frontal_lateral"],
            "ap_pa": row["ap_pa"],
        }

        return {
            "image": image_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "label_mask": label_mask,
            "metadata": metadata,
        }
