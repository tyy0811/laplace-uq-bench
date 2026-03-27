"""8-channel BC conditioning tensor encoding.

Channels 0-3: BC values broadcast along perpendicular axis.
  0 (top):    bc_top[j] broadcast to every row i
  1 (bottom): bc_bottom[j] broadcast to every row i
  2 (left):   bc_left[i] broadcast to every column j
  3 (right):  bc_right[i] broadcast to every column j

Channels 4-7: observation masks with same broadcast pattern.
  Phase 1: all 1.0 (fully observed exact BCs).
  Phase 2: 1.0 at observed positions, 0.0 at interpolated.
"""

import torch


def encode_conditioning(
    bc_top, bc_bottom, bc_left, bc_right,
    mask_top=None, mask_bottom=None, mask_left=None, mask_right=None,
):
    """Build (8, nx, nx) conditioning tensor from boundary arrays.

    Args:
        bc_top, bc_bottom: (nx,) tensors, broadcast along rows.
        bc_left, bc_right: (nx,) tensors, broadcast along columns.
        mask_*: optional (nx,) tensors; default all 1.0.

    Returns:
        (8, nx, nx) float32 tensor.
    """
    nx = bc_top.shape[0]
    cond = torch.zeros(8, nx, nx, dtype=torch.float32)

    # Value channels
    cond[0] = bc_top.unsqueeze(0).expand(nx, nx)
    cond[1] = bc_bottom.unsqueeze(0).expand(nx, nx)
    cond[2] = bc_left.unsqueeze(1).expand(nx, nx)
    cond[3] = bc_right.unsqueeze(1).expand(nx, nx)

    # Mask channels
    if mask_top is None:
        cond[4:8] = 1.0
    else:
        cond[4] = mask_top.unsqueeze(0).expand(nx, nx)
        cond[5] = mask_bottom.unsqueeze(0).expand(nx, nx)
        cond[6] = mask_left.unsqueeze(1).expand(nx, nx)
        cond[7] = mask_right.unsqueeze(1).expand(nx, nx)

    return cond
