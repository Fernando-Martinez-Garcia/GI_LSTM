import numpy as np
import torch
import torch.nn as nn
import json
import time
import os

from gilstm import GI_LSTM
from dataset import train_loader, test_loader, X_test_tensor, y_test_tensor, scaler_y


with open("parameters.json") as file:
    parameters = json.load(file)


input_size = parameters["input_size"]
hidden_size = parameters["hidden_size"]
output_size = parameters["output_size"]
qs=parameters["memory_group_shape"]
num_epochs = parameters["epoch_number"]
device_used=torch.device(parameters["device"])

torch.set_default_device(device_used)


model = GI_LSTM(input_size, hidden_size, output_size, qs=qs,device=device_used).to(device_used)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
training_record=[parameters]


for epoch in range(num_epochs):
    start_time=time.time()
    model.train()
    epoch_loss = 0
    

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device_used)
        batch_y = batch_y.to(device_used)

        batch_x = batch_x.unsqueeze(2)

        optimizer.zero_grad()

        output = model(batch_x)
        last_output = output[:, -1, :]

        loss = criterion(last_output, batch_y)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_x.size(0)

    avg_train_loss = epoch_loss / len(train_loader.dataset)

    model.eval()
    test_loss = 0
    difference = 0
    count = 0
    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_x = test_x.to(device_used)
            test_y = test_y.to(device_used)

            test_x = test_x.unsqueeze(2)

            output = model(test_x)
            last_output = output[:, -1, :]

            loss = criterion(last_output, test_y)

            y_pred=scaler_y.inverse_transform(last_output.cpu().detach().numpy())
            y_test=scaler_y.inverse_transform(test_y.cpu().detach().numpy())
            difference += np.mean(np.abs(y_pred - y_test))
            count += 1

            test_loss += loss.item() * test_x.size(0)

    avg_test_loss = test_loss / len(test_loader.dataset)
    ave_difference= difference / count

    time_taken=time.time()-start_time
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}, Test Difference {ave_difference}, Time Elapsed: {time_taken}')
    record=[num_epochs,epoch,time_taken,avg_train_loss,avg_test_loss,ave_difference.tolist()]
    training_record.append(record)


model_path=os.path.join(os.getcwd(),"models",f"model_d_{parameters["dataset_name"]}_h_{hidden_size}_qs_{str(model.qs)}.pth")
torch.save(model.state_dict(),model_path)
model = GI_LSTM(input_size, hidden_size, output_size, qs=qs,device=device_used)
model.load_state_dict(torch.load(model_path,weights_only=True))



model.eval()
with torch.no_grad():
    X_test_eval = X_test_tensor.to(device_used)
    y_test_eval = y_test_tensor.to(device_used)

    X_test_eval = X_test_eval.unsqueeze(2)

    output = model(X_test_eval)
    last_output = output[:, -1, :]
    eval_loss = criterion(last_output, y_test_eval)
    print(f'Final Test Loss: {eval_loss.item():.6f}')

    y_pred_scaled = last_output.cpu().numpy()
    y_test_scaled = y_test_eval.cpu().numpy()

    y_pred_original = scaler_y.inverse_transform(y_pred_scaled)
    y_test_original = scaler_y.inverse_transform(y_test_scaled)

    mse_original = np.mean((y_pred_original - y_test_original) ** 2)
    print(f'Final Test MSE on Original Scale: {mse_original:.6f}')
    training_record.append([mse_original.tolist()])
    with open(f'{model_path}.json',"w") as file:
        json.dump(training_record,file)
