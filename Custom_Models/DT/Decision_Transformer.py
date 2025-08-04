import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Requirements:
    Discern between different types of inputs such as images, vectors etc.
    Understand timeseries
    Tokenize those values so they can be processed later down the line
"""


class StateEmbedder(nn.Module):
    def __init__(self,max_input_dim, embed_dim):
        super(StateEmbedder,self).__init__()
        self.max_input_dim = max_input_dim
        self.embed_dim = embed_dim
        self.projection = nn.Linear(max_input_dim,embed_dim)
    def forward(self,x):
        return self.projection(x)

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size=128, max_state_dim = 20, max_length=20, n_layers=3, n_heads=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_state_dim = max_state_dim
        self.max_length = max_length
        # Input embedding
        self.embed_timestep = nn.Embedding(max_length, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state_vector = nn.Linear(max_state_dim, hidden_size)
        self.embed_state_image = nn.Conv2d(3, hidden_size, kernel_size=3, stride=1, padding=1)
        self.embed_action = nn.Linear(act_dim, hidden_size)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Prediction heads
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, act_dim)
        )

    def forward(self, states, prev_actions, returns_to_go, timesteps):
        """
        Inputs:
            states:        [B, T, state_dim]
            actions:       [B, T, act_dim]
            returns_to_go: [B, T, 1]
            timesteps:     [B, T]
        Output:
            action_preds:  [B, T, act_dim]
        """
        B, T = states.shape[0], states.shape[1]

        # Embed each modality
        time_emb = self.embed_timestep(timesteps)                       # (B, T, hidden)
        states,mask = self.padding(states)
        state_emb = self.embed_state_vector(states)                 # (B, T, hidden)
        print(state_emb.size())
        print(time_emb.size())
        state_emb = state_emb + time_emb
        action_emb = self.embed_action(prev_actions) + time_emb             # (B, T, hidden)
        return_emb = self.embed_return(returns_to_go) + time_emb       # (B, T, hidden)

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