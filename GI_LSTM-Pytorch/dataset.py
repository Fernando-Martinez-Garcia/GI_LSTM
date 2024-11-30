from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
import os

with open("parameters.json") as file:
    parameters = json.load(file)

device = torch.device(parameters["device"])
batch_size = parameters["batch_size"]
seq_len = parameters["sequence_length"]
dataset_name=parameters["dataset_name"]

torch.set_default_device(device)

with open(os.path.join(os.getcwd(),"dataset",dataset_name)) as file:

    data = file.read().splitlines()
    data = list(map(lambda x: float(x), data))
    data = np.array(data)

def create_sequences(data, seq_len):
    xs = []
    ys = []
    for i in range(len(data) - seq_len):
        x = data[i:(i + seq_len)]
        y = data[i + seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(data, seq_len)

train_size = int(len(X) * 0.8)

X_train_np = X[:train_size]
y_train_np = y[:train_size]
X_test_np = X[train_size:]
y_test_np = y[train_size:]

scaler_X = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(-1, 1))

X_train = scaler_X.fit_transform(X_train_np)
y_train = y_train_np.reshape(-1, 1)
y_train = scaler_y.fit_transform(y_train)

X_test = scaler_X.transform(X_test_np)
y_test = y_test_np.reshape(-1, 1)
y_test = scaler_y.transform(y_test)


X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
