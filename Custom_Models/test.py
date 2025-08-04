import pandas as pd
import numpy as np
from DT.Decision_Transformer import DecisionTransformer as DT
import torch
import torch.optim as optim
import torch.nn as nn
import ast

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
#model = DT(8, 2,max_length=5001)
#optimizer = optim.Adam(model.parameters(), lr=0.001)
#criterion = nn.MSELoss()
#for i in range(5):
#    pred = model(ep1_3DBall_obs,ep1_3DBall_act,ep1_3DBall_rew,ep1_3DBall_tim)
    