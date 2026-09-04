import json
from pathlib import Path
import numpy as np
import pandas as pd

TARGET_COLUMNS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

ALL_LABEL_COLUMNS = TARGET_COLUMNS + ["No Finding"]

def load_label_json(path: str | Path) -> pd.DataFrame:
    """
    Load a line-delimited JSON label file.
    Each line contains one JSON object corresponding to one image.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return pd.DataFrame(records)

def apply_label_policy(
    labels_df: pd.DataFrame,
    target_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply the primary uncertainty policy (U-Ignore).

    Label semantics:
        1.0  -> positive
        0.0  -> negative
        -1.0 -> uncertain -> ignored
        NaN  -> missing   -> ignored

    Returns:
        targets: DataFrame containing 0/1 targets and NaN for ignored labels
        masks:   DataFrame containing 1 where the label contributes to loss,
                 0 where the label is ignored
    """
    if target_columns is None:
        target_columns = TARGET_COLUMNS

    targets = pd.DataFrame(index=labels_df.index)
    masks = pd.DataFrame(index=labels_df.index)

    for label in target_columns:
        values = labels_df[label]

        # Only explicit positive (1.0) and negative (0.0) labels are used for supervision
        targets[label] = values.where(
            values.isin([0.0, 1.0]),
            np.nan,
        )

        # 1 = valid supervised target, 0 = uncertain (-1.0) or missing (NaN)
        masks[label] = values.isin([0.0, 1.0]).astype(np.int8)

    return targets, masks
