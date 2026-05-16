"""
Skeleton retargeting for NTU 25-joint data.
Replaces every bone with the dataset's average length, but keeps the
direction the bone was pointing in that frame.
"""

import torch


# NTU 25-joint parent map. -1 means root (no parent).
NTU_PARENTS = [
    -1,  # 0  spine base
     0,  # 1  spine middle
    20,  # 2  neck
     2,  # 3  head
    20,  # 4  left shoulder
     4,  # 5  left elbow
     5,  # 6  left wrist
     6,  # 7  left hand
    20,  # 8  right shoulder
     8,  # 9  right elbow
     9,  # 10 right wrist
    10,  # 11 right hand
     0,  # 12 left hip
    12,  # 13 left knee
    13,  # 14 left ankle
    14,  # 15 left foot
     0,  # 16 right hip
    16,  # 17 right knee
    17,  # 18 right ankle
    18,  # 19 right foot
     1,  # 20 spine shoulder
     7,  # 21 left hand tip
     7,  # 22 left thumb
    11,  # 23 right hand tip
    11,  # 24 right thumb
]

# Walk order: parents come before children, so when we place a child joint
# its parent has already been placed in the output.
NTU_TOPO_ORDER = [
    0, 1, 20, 2, 3, 4, 5, 6, 7, 21, 22,
    8, 9, 10, 11, 23, 24, 12, 13, 14, 15,
    16, 17, 18, 19,
]


def compute_canonical_bone_lengths(skeletons, parents=NTU_PARENTS, eps=1e-6):
    """Mean length for each bone across all valid frames in the dataset."""
    x = skeletons.float()
    V = x.shape[3]

    # The NTU recordings are zero-padded out to T=300 frames, so a lot of
    # the late frames are all zeros. We detect those by checking whether
    # the root joint has any non-zero coordinate, and skip them.
    root = x[:, :, :, 0, :]
    valid = root.abs().sum(dim=1) > eps

    canonical = torch.zeros(V, device=x.device)
    for v in range(V):
        p = parents[v]
        if p == -1:
            # root has no parent, so no bone to measure
            continue
        # bone vector = child position - parent position. We want its length.
        bone = x[:, :, :, v, :] - x[:, :, :, p, :]
        length = bone.norm(dim=1)
        # only average over frames that are valid AND where the bone actually
        # has some length (degenerate bones near zero get filtered out)
        good = valid & (length > eps)
        if good.any():
            canonical[v] = length[good].mean()
    return canonical


def retarget_skeletons(skeletons, parents=NTU_PARENTS, topo_order=NTU_TOPO_ORDER,
                       canonical_lengths=None, eps=1e-6):
    """Walk the skeleton and rewrite every bone to its canonical length."""
    if canonical_lengths is None:
        canonical_lengths = compute_canonical_bone_lengths(skeletons, parents, eps)

    orig_dtype = skeletons.dtype
    x = skeletons.float()

    # Output starts as zeros and we fill it in joint-by-joint. The root
    # joint stays exactly where it was so the person's overall position in
    # the room is preserved.
    out = torch.zeros_like(x)
    out[:, :, :, 0, :] = x[:, :, :, 0, :]

    # Same frame validity check as in compute_canonical_bone_lengths.
    root = x[:, :, :, 0, :]
    valid = (root.abs().sum(dim=1) > eps).unsqueeze(1)

    canonical_lengths = canonical_lengths.to(x.device)

    # For each child joint, walked in parent-first order:
    #   1. Get the bone vector from the parent to the child in the original.
    #   2. Normalize it to a unit direction.
    #   3. Place the child at parent_position + canonical_length * direction.
    # Because we walk parents before children, the parent is always already
    # placed in `out` by the time we get here.
    for v in topo_order:
        p = parents[v]
        if p == -1:
            continue
        bone = x[:, :, :, v, :] - x[:, :, :, p, :]
        length = bone.norm(dim=1, keepdim=True)
        direction = bone / length.clamp(min=eps)
        # zero out direction on padded/degenerate frames so we don't
        # propagate garbage out from a bad input frame
        keep = (length > eps) & valid
        direction = direction * keep.float()
        out[:, :, :, v, :] = out[:, :, :, p, :] + canonical_lengths[v] * direction

    return out.to(orig_dtype)
