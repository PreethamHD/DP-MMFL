import torch
import torch.nn.functional as F


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes element-wise Binary Cross-Entropy with Logits, gates loss elements
    by binary mask (U-Ignore policy), and normalizes by the count of observed labels.
    """
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    masked_loss = loss * mask
    valid_count = mask.sum()

    if valid_count == 0:
        return masked_loss.sum() * 0.0

    return masked_loss.sum() / valid_count
