import pandas as pd
import numpy as np
from DT.Decision_Transformer import DecisionTransformer as DT
import torch
import torch.optim as optim
import torch.nn as nn
import ast
import pickle
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
pickle_path = "MultiEnvironmentProject/Database/Racing_Test_dataset.pkl"
try:
    with open(pickle_path, 'rb') as f:
        dataset = pickle.load(f)
except FileNotFoundError:
    print(f"Dataset file not found at: {pickle_path}")
    raise
except Exception as e:
    print(f"Error loading pickle file: {e}")
    raise

#df = pickle.load("MultiEnvironmentProject/Database/Racing_Test_dataset.pkl")
print(len(dataset["observations"][0]))
observations = torch.tensor(np.array(dataset["observations"]),dtype=torch.float32).to(device)
print("Observation size:",observations.size())
actions = torch.tensor(np.array(dataset["actions"]),dtype=torch.float32).to(device)
rewards = torch.tensor(np.array(dataset["rewards"]),dtype=torch.float32).to(device).view(30,600,1)
dones = torch.tensor(np.array(dataset["dones"]),dtype=torch.int32).to(device)
timewindow = torch.tensor([[i for i in range(rewards.shape[1])] for _ in range(rewards.shape[0])],dtype=torch.int32).to(device)
# Number of training steps
num_epochs = 5
context_len = 30  # window size
horizon = 1       # predict 1 step ahead

model = DT(8, 2, max_state_dim=100, max_action_dim=5, max_length=context_len, device=device).to(device)  # adjust according to your model
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
obs_windows = torch.split(observations, 30, dim=1)
act_windows = torch.split(actions, 30, dim=1)
rew_windows = torch.split(rewards, 30, dim=1)
tim_windows = torch.split(timewindow, 30, dim=1)
don_windows = torch.split(dones, 30, dim=1)

for epoch in range(num_epochs):
    total_loss = 0.0
    count = 0

    for i in range(len(obs_windows)):
        # Slice input windows
        obs_window = obs_windows[i]
        act_window = act_windows[i]
        rew_window = rew_windows[i]
        print(rew_window.size())
        tim_window = tim_windows[i]
        don_window = don_windows[i]
        print("Obs window:",obs_window.size())
        # Target action at t + context_len
        target = act_window
        # Forward pass
        pred = model(obs_window, act_window, rew_window, tim_window, don_window)
        print("PREDICTIONS:",pred.size(),"TARGET:",target.size())
        print(pred.device, target.device)
        print(pred)
        print(target)
        # Compute loss (compare only the last predicted action)
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(loss)
        #total_loss += loss.item()
        count += 1

    print(f"Epoch {epoch+1}: avg loss = {total_loss / count:.4f}")