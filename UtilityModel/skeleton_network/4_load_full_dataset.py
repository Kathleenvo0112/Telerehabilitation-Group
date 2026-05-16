import os
import numpy as np
import torch
from tqdm import tqdm

def read_skeleton_file(file_path, max_people=2, num_joints=25):

    # number of joints per skeleton and dimensions (xyz)
    V = num_joints
    C = 3

    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    # number of frames
    T = int(lines[0])

    skeleton_data = []
    idx = 1
    for t in range(T):
        # M is the number of people per frame
        M = int(lines[idx])
        idx += 1
        frame_people = []

        for m in range(M):
            # skip metadata info
            idx += 1  

            # number of joints for this person
            v_count = int(lines[idx])
            idx += 1
            joints = []

            for v in range(v_count):
                # gets xyz coordinates, adds them to joints array
                joint_line = lines[idx].split()
                x, y, z = map(float, joint_line[:3])

                joints.append([x, y, z])
                idx += 1

            # full skeleton of person
            frame_people.append(np.array(joints))


        # if people is less than 2, add a second person to keep dimensions consistent
        while len(frame_people) < max_people:
            frame_people.append(np.zeros((V, C)))

        frame_people = frame_people[:max_people]

        # gets full tensor for frame [people, joints, xyz]
        skeleton_data.append(np.stack(frame_people, axis=0))

    # gets full tensor for all frames [people, frames, joints, xyz]
    return np.stack(skeleton_data, axis=1)  # [M, T, V, C]


if __name__ == "__main__":
    skeleton_dir = r"C:\Users\avido\Projects\rehab_data\nturgb+d_skeletons"
    MAX_T = 300 # max number of frames allowed
    MAX_M = 2 # max number of people allowed
    V = 25 # number of joints
    C = 3 # number of coordinates (xyz)

    file_list = [f for f in os.listdir(skeleton_dir) if f.endswith(".skeleton")]
    N = len(file_list)
    print(f"Found {N} skeleton files.")

    # create arrays
    skeletons_np = np.zeros((N, MAX_M, MAX_T, V, C), dtype=np.float32)
    labels_np = np.zeros(N, dtype=np.int64)


    for i, file_name in enumerate(tqdm(file_list, desc="Loading skeletons")):
        file_path = os.path.join(skeleton_dir, file_name)

        # Extract action label: S001C001P001R001A001 -> 1
        action_id = int(file_name.split('A')[1].split('.')[0])

        joints = read_skeleton_file(file_path, max_people=MAX_M)

        # get tensor dimensions for current file
        M, T, V_check, C_check = joints.shape

        # pad or truncate frames so everything is 300 frames
        if T < MAX_T:
            pad = np.zeros((M, MAX_T - T, V, C))
            joints = np.concatenate([joints, pad], axis=1)
        elif T > MAX_T:
            joints = joints[:, :MAX_T, :, :]

        # stores skeleton data and label
        skeletons_np[i] = joints
        labels_np[i] = action_id

    # convert to PyTorch tensors and reorder to [N, C, T, V, M]
    skeletons = torch.tensor(skeletons_np, dtype=torch.float32).permute(0, 4, 2, 3, 1)
    labels = torch.tensor(labels_np, dtype=torch.long)

    print("Skeleton dataset shape:", skeletons.shape)
    print("Labels shape:", labels.shape)

    # save
    torch.save({'skeletons': skeletons, 'labels': labels}, 'ntu_skeleton_dataset.pt')
    print("Saved dataset to ntu_skeleton_dataset.pt")