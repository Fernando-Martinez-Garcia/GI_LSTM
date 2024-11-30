import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import os
import json

device=torch.device("mps")
time_steps = 24

input_size = 1
hidden_size = 128
output_size = 1
num_layers = 1
learning_rate = 0.001
num_epochs = 100
batch_size = 10000
dataset_name="Binary_sequence_length28_28.txt"

with open(os.path.join(os.getcwd(),"dataset",dataset_name)) as file:

    data = file.read().splitlines()
    data = list(map(lambda x: float(x), data))

data=np.array(data)


values = data.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values)

def create_sequences(data, time_steps):
    sequences = []
    targets = []
    for i in range(len(data) - time_steps):
        sequences.append(data[i:i+time_steps])
        targets.append(data[i+time_steps])
    return np.array(sequences), np.array(targets)


X, y = create_sequences(scaled_data, time_steps)



X_train=X[:int(len(X)*0.8)]
y_train=y[:int(len(X)*0.8)]

X_test=X[int(len(X)*0.8):]
y_test=y[int(len(X)*0.8):]


X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=1,device=device):
        super(LSTMModel, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size).to(device)

        self.device=device
    
    def forward(self, x):

        lstm_out, _ = self.lstm(x)
        
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        
        return out

model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pytorch_total_params = sum(p.numel() for p in model.parameters())
    
print(f'Total number of trainable parameters: {pytorch_total_params}')


train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset=TensorDataset(X_test,y_test)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

train_record=[]

for epoch in range(num_epochs):
    epoch_loss=0
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        
        optimizer.step()
            
        epoch_loss+=loss.item()*batch_x.size(0)

    avg_train_loss = epoch_loss / len(train_loader.dataset)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_x = test_x.to(device)
            test_y = test_y.to(device)
                
            output = model(test_x)
                
            loss = criterion(output, test_y)

            test_loss += loss.item() * test_x.size(0)
    
    avg_test_loss = test_loss / len(test_loader.dataset)

    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}' )
    train_record.append(avg_test_loss)


with open(f'{dataset_name}_training_record3.txt',"w") as file:
    json.dump(train_record,file)
