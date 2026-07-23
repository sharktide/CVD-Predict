import os
import torch
from safetensors.torch import save_file

# Define your target directory path
model_dir = "models/cardiac_arrest_v4"

# Get all .pt weight files in the directory
pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]

if not pt_files:
    print(f"No .pt files found in {model_dir}")
else:
    print(f"Found {len(pt_files)} PyTorch checkpoints to convert...\n")

for file_name in pt_files:
    pt_path = os.path.join(model_dir, file_name)
    sf_path = os.path.join(model_dir, file_name.replace('.pt', '.safetensors'))
    
    print(f"Processing: {file_name}...")
    
    try:
        # Load the PyTorch checkpoint securely using weights_only=True
        checkpoint = torch.load(pt_path, map_location="cpu", weights_only=True)
        
        # Check if weights are nested (common with training scripts)
        if isinstance(checkpoint, dict):
            # Extract state dict if nested under a common key
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            # If the file saved the entire raw model object instead of a dict
            print(f"  [Warning] {file_name} contains a full object, attempting state_dict extraction...")
            state_dict = checkpoint.state_dict()

        # Safetensors requires tensor keys to be strings and continuous
        # This converts any potential non-contiguous tensors to contiguous layout
        clean_state_dict = {k: v.contiguous() for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
        
        # Save to the new format
        save_file(clean_state_dict, sf_path)
        print(f"  [Success] Saved to: {os.path.basename(sf_path)}")
        
    except Exception as e:
        print(f"  [Error] Failed to convert {file_name}: {e}")

print("\nBatch conversion process finished.")
