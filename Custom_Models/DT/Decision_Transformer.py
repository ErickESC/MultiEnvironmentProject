import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...Agents.DT.multiEnvDT import MultiGameDTConfig,GameConfig
import cv2
"""
Requirements:
    Discern between different types of inputs such as images, vectors etc.
    Understand timeseries
    Tokenize those values so they can be processed later down the line
"""


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size=128, max_state_dim = 20, max_action_dim=50, max_length=20, n_layers=3, n_heads=4,device="cpu"):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.max_state_dim = max_state_dim
        self.max_length = max_length
        self.max_action_dim = max_action_dim
        # Input embedding
        self.embed_timestep = nn.Embedding(max_length, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state_vector = nn.Linear(max_state_dim, hidden_size)
        self.embed_state_image = nn.Conv2d(3, hidden_size, kernel_size=3, stride=1, padding=1)
        self.embed_action = nn.Linear(max_action_dim, hidden_size)
        self.emben_dones = nn.Linear(max_length,hidden_size)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Prediction heads
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, max_action_dim)
        )

    def forward(self, states, prev_actions, returns_to_go, timesteps, dones):
        """
        Inputs:
            states:        [B, T, state_dim]
            actions:       [B, T, act_dim]
            returns_to_go: [B, T, 1]
            dones:         [B, T]
            timesteps:     [B, T]
        Output:
            action_preds:  [B, T, act_dim]
        """
        B, T = states.shape[0], states.shape[1]
        A_B = prev_actions.shape[2]
        #print(states.device,prev_actions.device, returns_to_go.device, timesteps.device, dones.device)
        # Embed each modality
        time_emb = self.embed_timestep(timesteps)                       # (B, T, hidden)
        states = self.padding(T,states,pad_size=self.max_state_dim)
        state_emb = self.embed_state_vector(states)                 # (B, T, hidden)
        state_emb = state_emb + time_emb
        action_emb = self.embed_action(self.padding(T,prev_actions,self.max_action_dim)) + time_emb             # (B, T, hidden)
        return_emb = self.embed_return(returns_to_go) + time_emb       # (B, T, hidden)
        #dones_emb = self.emben_dones(self.padding(T,dones))
        
        # Stack into sequence: [r_1, s_1, a_1, r_2, s_2, a_2, ..., r_T, s_T, a_T]
        stacked_inputs = torch.stack(
            (return_emb, state_emb, action_emb), dim=2
        ).reshape(B, 3 * T, self.hidden_size)

        # Transformer
        x = self.transformer(stacked_inputs)                            # (B, 3T, hidden)

        # Extract only action prediction positions
        x = x.reshape(B, T, 3, self.hidden_size)                        # (B, T, 3, hidden)
        action_tokens = x[:, :, 2, :]                                   # (B, T, hidden)
        action_preds = self.predict_action(action_tokens)              # (B, T, act_dim)

        return action_preds
    def padding(self,input_tensor:torch.Tensor):
        if input_tensor.size(-1) < self.max_state_dim:
            zeros = torch.zeros(input_tensor.size(0), input_tensor.size(1), self.max_state_dim  - input_tensor.size(2))
            output_tensor = torch.cat((input_tensor,zeros), dim=2)
            mask = torch.zeros(output_tensor.size(0),output_tensor.size(1),output_tensor.size(2))
            mask[:input_tensor.size(0), :input_tensor.size(1), :input_tensor.size(2)] = 1
            return output_tensor,mask
        mask = torch.ones(input_tensor.size(0), input_tensor.size(1), input_tensor.size(2))
        return input_tensor,mask