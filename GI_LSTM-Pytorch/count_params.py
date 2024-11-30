from gilstm import GI_LSTM
import torch
import json
import os

with open("parameters.json") as file:
    parameters = json.load(file)


input_size = parameters["input_size"]
hidden_size = parameters["hidden_size"]
output_size = parameters["output_size"]
qs=parameters["memory_group_shape"]
num_epochs = parameters["epoch_number"]
device_used=torch.device(parameters["device"])

model = GI_LSTM(input_size, hidden_size, output_size, qs=qs,device=device_used).to(device_used)
# model_path=os.path.join(os.getcwd(),"models",f"model_d_{parameters["dataset_name"]}_h_{hidden_size}_qs_{str(model.qs)}.pth")
# model.load_state_dict(torch.load(model_path,weights_only=True))

pytorch_total_params = sum(p.numel() for p in model.parameters())

print(f'Total number of trainable parameters: {pytorch_total_params}')
