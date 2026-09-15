import torch
from model import DownscalingSDE

state_dict = torch.load('./checkpoints/cfm/best_model_phase1.pt', map_location=torch.device('cpu'))

model = DownscalingSDE() 
model.load_state_dict(state_dict)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")