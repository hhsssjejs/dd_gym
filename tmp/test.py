import numpy as np
import torch
import matplotlib.pyplot as plt


# Define parameters
num_phases = 100  # Number of phase samples
device = "cpu"

# Generate phase values from 0 to 1
phase = torch.linspace(0, 10, num_phases, device=device)
sin_pos = torch.sin(2 * torch.pi * phase)

# Compute stance mask
stance_mask = torch.zeros((num_phases, 2), device=device)
stance_mask[:, 0] = sin_pos >= 0  # Left foot stance
stance_mask[:, 1] = sin_pos < 0   # Right foot stance
stance_mask[torch.abs(sin_pos) < 0.05] = 1  # Double support phase

# Convert to NumPy for plotting
phase_np = phase.cpu().numpy()
stance_left = stance_mask[:, 0].cpu().numpy()
stance_right = stance_mask[:, 1].cpu().numpy()

# Plot stance phases
plt.figure(figsize=(10, 5))
plt.plot(phase_np, stance_left, label="Left Foot Stance", drawstyle="steps-post")
plt.plot(phase_np, stance_right, label="Right Foot Stance", drawstyle="steps-post")
plt.fill_between(phase_np, 0, stance_left, step="post", alpha=0.3)
plt.fill_between(phase_np, 0, stance_right, step="post", alpha=0.3, color="orange")

plt.xlabel("Phase")
plt.ylabel("Stance Mask")
plt.title("Left and Right Foot Stance Phases")
plt.legend()
plt.grid()
plt.show()