import pandas as pd
import numpy as np
from DT.Decision_Transformer import DecisionTransformer as DT
import torch
import torch.optim as optim
import torch.nn as nn
import ast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv("MultiEnvironmentProject/Database/MLAgentsOutputs.csv")
print(df.keys())
ep1_3DBall = df[df['episode'] == 1]
ep1_3DBall = ep1_3DBall[ep1_3DBall["game"]=="3DBall"]
ep1_3DBall = ep1_3DBall[["observations","actions","rewards","time_step"]]
ep1_3DBall_obs = torch.tensor([ast.literal_eval(ep1_3DBall["observations"][i]) for i in range(len(ep1_3DBall["observations"]))],dtype=torch.float32).view(1, 5001, 8)
ep1_3DBall_act = torch.tensor([ast.literal_eval(ep1_3DBall["actions"][i]) for i in range(len(ep1_3DBall["actions"]))],dtype=torch.float32).view(1, 5001, 2)
ep1_3DBall_rew = torch.tensor([[[ep1_3DBall["rewards"][i]] for i in range(len(ep1_3DBall["rewards"]))]],dtype=torch.float32)
ep1_3DBall_tim = torch.tensor([[ep1_3DBall["time_step"][i] for i in range(len(ep1_3DBall["time_step"]))]],dtype=torch.int32)

print(ep1_3DBall_obs.size())
print(ep1_3DBall_act.size())
print(ep1_3DBall_rew.size())
print(ep1_3DBall_tim.size())
print(ep1_3DBall_obs.dtype)
context_len = 20  # window size
horizon = 1       # predict 1 step ahead

model = DT(8, 2, max_length=5001)  # adjust according to your model
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Number of training steps
num_epochs = 5

for epoch in range(num_epochs):
    total_loss = 0.0
    count = 0

    for t in range(0, ep1_3DBall_obs.shape[0] - context_len - horizon, 1):
        # Slice input windows
        obs_window = ep1_3DBall_obs[t : t + context_len]
        act_window = ep1_3DBall_act[t : t + context_len]
        rew_window = ep1_3DBall_rew[t : t + context_len]
        tim_window = ep1_3DBall_tim[t : t + context_len]

        # Target action at t + context_len
        target = ep1_3DBall_act[t + context_len : t + context_len + horizon]


        # Add batch dimension (1, context_len, feature_dim)
        obs_window = obs_window.unsqueeze(0)
        act_window = act_window.unsqueeze(0)
        rew_window = rew_window.unsqueeze(0)
        tim_window = tim_window.unsqueeze(0)
        target = target.unsqueeze(0)

        # Forward pass
        pred = model(obs_window, act_window, rew_window, tim_window)

        # Compute loss (compare only the last predicted action)
        loss = criterion(pred[:, -horizon:], target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    print(f"Epoch {epoch+1}: avg loss = {total_loss / count:.4f}")