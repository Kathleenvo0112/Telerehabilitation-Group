import torch
import numpy as np
from tqdm import tqdm

#parent joint labeled via diagram in NTU paper
#-1 at joint root, has no parent
NTU_PARENTS = [
    -1,  #0  spine base
     0,  #1  spine middle
    20,  #2  neck
     2,  #3  head
    20,  #4  left shoulder
     4,  #5  left elbow
     5,  #6  left wrist
     6,  #7  left hand
    20,  #8  right shoulder
     8,  #9  right elbow
     9,  #10 right wrist
    10,  #11 right hand
     0,  #12 left hip
    12,  #13 left knee
    13,  #14 left ankle
    14,  #15 left foot
     0,  #16 right hip
    16,  #17 right knee
    17,  #18 right ankle
    18,  #19 right foot
     1,  #20 spine shoulder
     7,  #21 left hand tip
     7,  #22 left thumb
    11,  #23 right hand tip
    11,  #24 right thumb
]

#order of processing joints from root to outward to ensure parents are processed before children
JOINT_ORDER = [
    0, 1, 20, 2, 3, 4, 5, 6, 7, 21, 22,
    8, 9, 10, 11, 23, 24, 12, 13, 14, 15,
    16, 17, 18, 19,
]


def quantize_angles(skeletons, bin_deg=10, batch_size=200, eps=1e-6):
    #converts the bin size from deg to rad
    bin_rad = np.deg2rad(bin_deg)
    N, C, T, V, M = skeletons.shape
    print(f"Angle quantization: bin size {bin_deg} degrees")

    output = torch.zeros_like(skeletons)

    for i in tqdm(range(0, N, batch_size), desc="Quantizing angles"):
        end = min(i + batch_size, N)
        x = skeletons[i:end].float()

        out = torch.zeros_like(x)
        #keep root joint unchanged to preserve global position
        out[:, :, :, 0, :] = x[:, :, :, 0, :]

        #check which frames are valid based on root joint movement
        valid = (x[:, :, :, 0, :].abs().sum(dim=1, keepdim=True) > eps)

        for v in JOINT_ORDER:
            p = NTU_PARENTS[v]
            #skip root joint since no parent
            if p == -1:
                continue

            #computing bone vector from parent to child joint
            bone = x[:, :, :, v, :] - x[:, :, :, p, :]
            #getting bone length and normalize
            length = bone.norm(dim=1, keepdim=True).clamp(min=eps)
            direction = bone / length

            #convert to XYZ components
            dx = direction[:, 0, :, :]
            dy = direction[:, 1, :, :]
            dz = direction[:, 2, :, :]

            #convert to spherical angles
            #theta is the vertical
            #phi is horizontal
            theta = torch.asin(dy.clamp(-1 + eps, 1 - eps))
            # phi = azimuth angle in XZ plane [-pi, pi]
            phi = torch.atan2(dz, dx)

            #round angles to nearest bin
            theta_q = torch.round(theta / bin_rad) * bin_rad
            phi_q   = torch.round(phi   / bin_rad) * bin_rad

            #convert angles back to 3D vector
            cos_theta = torch.cos(theta_q)
            dx_q = cos_theta * torch.cos(phi_q)
            dy_q = torch.sin(theta_q)
            dz_q = cos_theta * torch.sin(phi_q)

            direction_q = torch.stack([dx_q, dy_q, dz_q], dim=1)

            #zero out invalid frames
            direction_q = direction_q * valid.float()

            #reconstruct joint position
            out[:, :, :, v, :] = out[:, :, :, p, :] + direction_q * length

        output[i:end] = out.to(skeletons.dtype)

    return output

def main():
    #change these settings as needed
    INPUT_FILE = "ntu_skeleton_dataset.pt"
    OUTPUT_FILE = "ntu_quantized.pt"
    BIN_DEG = 10
    BATCH_SIZE = 200

    print(f"Loading {INPUT_FILE}")
    data = torch.load(INPUT_FILE, map_location="cpu")
    skeletons = data["skeletons"]
    print(f"Loaded {skeletons.shape[0]} sequences, shape {list(skeletons.shape)}")

    #runs quantization
    quantized = quantize_angles(skeletons, bin_deg=BIN_DEG, batch_size=BATCH_SIZE)
    
    #save the quantized dataset
    data["skeletons"] = quantized
    torch.save(data, OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()