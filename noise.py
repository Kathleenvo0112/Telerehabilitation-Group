import torch

input_path = 'ntu_skeleton_dataset.pt'
output_path = 'ntu_noisy_0.1.pt'
noise_level=0.01

data = torch.load(input_path)
skeletons = data['skeletons']

noise = torch.randn_like(skeletons) * noise_level

# Apply noise
data['skeletons'] = skeletons + noise

torch.save(data, output_path)
print(f"Saved to {output_path}")