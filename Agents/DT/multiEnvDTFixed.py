import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments, PreTrainedModel, PretrainedConfig
import numpy as np
import pickle
import logging
import os
import cv2
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import random


# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DuelingArchitecture(nn.Module):
    def __init__(self,embed_dim,act_dim):
        super(DuelingArchitecture,self).__init__()
        self.fc_adv = nn.Linear(embed_dim, embed_dim)
        self.fc_value = nn.Linear(embed_dim, embed_dim)
        self.adv = nn.Linear(embed_dim,act_dim)
        self.value = nn.Linear(embed_dim,1)
    def forward(self,x):
        x_adv = F.relu(self.fc_adv(x))
        x_adv = self.adv(x_adv)
        x_value = F.relu(self.fc_value(x))
        x_value = self.value(x_value)
        adv_average = torch.mean(x_adv,dim=2,keepdim=True)
        return x_value + x_adv - adv_average

class DecisionTransformerWithImage(nn.Module):
    def __init__(self,
                config,
                dueling_network:bool = True):
        super(DecisionTransformerWithImage,self).__init__()
        self.embedding_dim = config["embedding_dim"]
        self.max_state_dim = config["max_state_dim"]
        self.context_length = config["context_length"]
        self.n_layer = config["n_layer"]
        self.n_head = config["n_head"]
        self.max_ep_length = config["max_ep_length"]
        self.max_act_dim = config['max_act_dim']
        self.max_dis_act_dims = config['max_dis_act_dims']
        self.img_dim_x = config['img_dim_x']
        self.img_dim_y = config['img_dim_y']
        self.vocab = {combo: idx for idx, combo in enumerate(list(itertools.product(*[range(s) for s in self.max_dis_act_dims])))}
        self.reverse_vocab = {idx: combo for combo, idx in self.vocab.items()}
        # Input embedding
        self.state_encoder = nn.Linear(self.max_state_dim, self.embedding_dim)
        self.state_encoder_image = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, self.embedding_dim, kernel_size=3, stride=1, padding=1))
        self.action_encoder = nn.Linear(config.game_configs['max_act_dim'], self.embedding_dim)
        self.action_encoder = nn.Embedding(np.prod(self.max_dis_act_dims),self.embedding_dim)
        self.rtg_encoder = nn.Linear(1, self.embedding_dim)
        self.timestep_encoder = nn.Embedding(self.max_ep_length, self.embedding_dim)
        self.embed_ln = nn.LayerNorm(self.embedding_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.n_head,
            dim_feedforward=4 * self.embedding_dim,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layer)
        self.predict_ln = nn.LayerNorm(self.embedding_dim)

        self.action_heads = DuelingArchitecture(self.embedding_dim,len(self.vocab)) if config["dueling_arch"] else nn.Linear(self.embedding_dim,self.vocab)

    def _create_action_head(self, game_config, dueling_network:bool = False):
        head_layers = nn.ModuleDict()
        if not dueling_network:
            if game_config.action_type in ['continuous', 'mixed']:
                head_layers['continuous'] = nn.Linear(self.embedding_dim, game_config.action_continuous_dim)
            if game_config.action_type in ['discrete', 'mixed']:
                head_layers['discrete'] = nn.ModuleList([
                    nn.Linear(self.embedding_dim, card) for card in game_config.action_discrete_cardinalities
                ])
        else:
            if game_config.action_type in ['continuous', 'mixed']:
                head_layers['continuous'] = DuelingArchitecture(self.embedding_dim, game_config.action_continuous_dim)
            if game_config.action_type in ['discrete', 'mixed']:
                head_layers['discrete'] = nn.ModuleList([
                    DuelingArchitecture(self.embedding_dim, card) for card in game_config.action_discrete_cardinalities
                ])

        return head_layers
    def _pad_features_old(self, input_tensors:torch.Tensor, tensor_type=None) -> tuple[torch.Tensor, torch.Tensor]:
        if tensor_type == "action":
            if self.max_act_dim > input_tensors.size(-1):
                input_tensors = F.pad(input_tensors, (0, self.max_act_dim - input_tensors.size(-1)))

        elif tensor_type =="state":
            if self.max_state_dim > input_tensors.size(-1):
                input_tensors = F.pad(input_tensors, (0, self.max_state_dim - input_tensors.size(-1)))
        else:
            raise ValueError("tensor_type is invalid")
        return input_tensors
    def _pad_features(self,input_tensor, tensor_type:str=None):

        if type(input_tensor) == list:
            if tensor_type=="state":
                if len(input_tensor) < self.max_state_dim:
                    input_tensor.extend([0 for _ in range((self.max_state_dim - len(input_tensor)))])
            elif tensor_type=="action":
                if len(input_tensor) < self.max_act_dim:
                    input_tensor.extend([0 for _ in range((self.max_act_dim - len(input_tensor)))])
            else:
                raise ValueError("Tensor type not given")
            
        elif type(input_tensor) == np.ndarray:
            if tensor_type=="state":
                if input_tensor.shape[-1] < self.max_state_dim:
                    input_tensor = np.pad(input_tensor, (0,self.max_state_dim - input_tensor.shape[0]), mode="constant", constant_values=0)
            elif tensor_type=="action":
                if input_tensor.shape[-1] < self.max_act_dim:
                    input_tensor = np.pad(input_tensor, (0,self.max_act_dim - input_tensor.shape[0]), mode="constant", constant_values=0)
            else:
                raise ValueError("Tensor type not given")

        elif type(input_tensor) == torch.Tensor:
            if tensor_type=="state":
                if input_tensor.size(-1) < self.max_state_dim:
                    input_tensors = F.pad(input_tensors, (0, self.max_state_dim - input_tensors.size(-1)))
            elif tensor_type=="action":
                if input_tensor.size(-1) < self.max_act_dim:
                    input_tensors = F.pad(input_tensors, (0, self.max_act_dim - input_tensors.size(-1)))
            else:
                raise ValueError("Tensor type not given")
        else:
            raise ValueError("Unrecognized input type")
        return input_tensor

    def _pad_sequences(self, input_tensors:list, tensor_type = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tensor types: "action", "state"
        """
        if tensor_type == "action":
            x = pad_sequence(input_tensors, batch_first=True, padding_value=0)
            mask = torch.cat([
                torch.ones(x.size(0), input_tensors.size(-1), dtype=torch.bool),
                torch.zeros(x.size(0), self.max_act_dim - input_tensors.size(-1), dtype=torch.bool)
            ], dim=1)
        elif tensor_type == "state":
            x = pad_sequence(input_tensors, batch_first=True, padding_value=0)
            print(self.max_state_dim,input_tensors.size(-1),self.max_state_dim - input_tensors.size(-1))
            mask = torch.cat([
                torch.ones(x.size(0), input_tensors.size(-1), dtype=torch.bool),
                torch.zeros(x.size(0), self.max_state_dim - input_tensors.size(-1), dtype=torch.bool)
            ], dim=1)
        else:
            raise ValueError("tensor_type is invalid")

        return x, mask


    def _rescale_image(self, img:np.ndarray) -> torch.Tensor:
        """ Rescale images to accepted resolution"""
        if img.shape[2] != self.img_dim_x or img.shape[3] != self.img_dim_y:
            resized_images = np.zeros((img.shape[0], img.shape[1], self.img_dim_x, self.img_dim_y, 3), dtype=np.int64)
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    resized_images[j, i] = cv2.resize(img[i, j], (self.img_dim_x, self.img_dim_y))
            
            return torch.from_numpy(resized_images).float().permute(0, 1, 4, 2, 3) / 255
        return torch.from_numpy(img).float().permute(0, 1, 4, 2, 3) / 255


    def forward(self, states, actions, returns_to_go, timesteps,horizon=1, game_ids=None, img:bool = False):
        """
        States and actions are expected to be a list of lists of np arrays eg.: [[np.arr([1,2,3])]]
        """
        #actions and states are a list of lists of numpy arrays
        #rewards is a list of lists of a list of a single int/float
        batch_size, seq_len = len(states), len(states[0])
        
        ##Transform states, actions, rtg, timesteps into a list of torch tensors
        #if(type(states) == list):
        #    print(len(states))
        #    states = [[torch.tensor(single_state,dtype=torch.float32) for single_state in state] for state in states]
        #    states = [[torch.tensor(self._pad_features(state,"state")) for state in seq_states] for seq_states in states]
        #    #states = [[torch.from_numpy(single_state) for single_state in state] for state in states]
        #    states = [torch.stack(state,dim=0) for state in states]
        #    states = torch.stack(states,dim=0)
#
#
        #if(type(actions) == list):
#
        #    actions = [[torch.tensor(list(action),dtype=torch.int32) if np.issubdtype(action.dtype, np.integer) 
        #                else torch.from_numpy(action.float())
        #                for action in seq_actions]
        #                for seq_actions in actions]
        #    print(self._pad_features(actions[0][0],tensor_type="action").tolist())
        #    actions = [[torch.tensor(self.vocab[tuple(self._pad_features(action,tensor_type="action").tolist())])for action in seq_actions] for seq_actions in actions]
        #    actions = [torch.stack(action,dim=0) for action in actions]
        #    actions = torch.stack(actions,dim=0)

        #Transform states, actions, rtg, timesteps into a list of torch tensors
        if(type(states) == list):
            print(len(states))
            #states = [[torch.tensor(single_state,dtype=torch.float32) for single_state in state] for state in states]
            states = [[torch.tensor(self._pad_features(state,"state")) for state in seq_states] for seq_states in states]
            #states = [[torch.from_numpy(single_state) for single_state in state] for state in states]
            states = [torch.stack(state,dim=0) for state in states]
            states = torch.stack(states,dim=0)


        if(type(actions) == list):

            #actions = [[torch.tensor(list(action),dtype=torch.int32) if np.issubdtype(action.dtype, np.integer) 
            #            else torch.from_numpy(action.float())
            #            for action in seq_actions]
            #            for seq_actions in actions]
            #print(self._pad_features(actions[0][0],tensor_type="action").tolist())
            actions = [[torch.tensor(self.vocab[tuple(self._pad_features(action,tensor_type="action").tolist())])for action in seq_actions] for seq_actions in actions]
            actions = [torch.stack(action,dim=0) for action in actions]
            actions = torch.stack(actions,dim=0)

        if(type(returns_to_go) == list):
            returns_to_go = torch.tensor(returns_to_go,dtype=torch.float32)


        if(type(timesteps) == list):
            timesteps = torch.tensor(timesteps, dtype=torch.int32)
        #Pad sequences if needed
        states_mask = torch.ones(states.size(0),states.size(1))
        actions_mask = torch.ones(actions.size(0),actions.size(1))
        rtg_mask = torch.ones(returns_to_go.size(0),returns_to_go.size(1))

        if img:
            states = self._rescale_image(states)
        else:
            states = self._pad_features(states, tensor_type="state")

        actions = self._pad_features(actions, tensor_type="action")
        
        # Embed all inputs
        if img:
            state_embeds = self.state_encoder_image(states)
        else: 
            state_embeds = self.state_encoder(states.float())
        action_embeds = self.action_encoder(actions)
        rtg_embeds = self.rtg_encoder(returns_to_go.float())
        time_embeds = self.timestep_encoder(timesteps)

        state_embeds += time_embeds
        state_embeds = state_embeds
        rtg_embeds += time_embeds
        rtg_embeds = rtg_embeds
        action_embeds += time_embeds

        # Assemble sequence
        stacked_inputs = torch.stack((rtg_embeds, state_embeds, action_embeds), dim=1)
        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_len, self.embedding_dim)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Create attention mask
        stacked_mask = torch.stack((rtg_mask, states_mask, actions_mask), dim=1)
        stacked_mask = stacked_mask.permute(0, 2, 1).reshape(batch_size, 3 * seq_len)
        inverted_attention_mask = stacked_mask == 0

        # Transformer forward pass
        x = self.transformer(stacked_inputs, src_key_padding_mask=inverted_attention_mask)
        x = self.predict_ln(x)

        # Reshape and get predictions
        x = x.reshape(batch_size, seq_len, 3, self.embedding_dim).permute(0, 2, 1, 3)
        state_preds = x[:,1]  # Predictions from state representations
        all_predictions = self.action_heads(state_preds)
        all_predictions = F.softmax(all_predictions[:,:horizon,:],dim=0).argmax(dim=2)
        decoded = [list(self.reverse_vocab[i.item()]) for i in all_predictions.view(-1)]
        return decoded

if __name__ == "__main__":
    CONTEXT_LENGTH = 20
    paths = ["MultiEnvironmentProject/Database/pirates_PPO_dataset.pkl"]
    with open("MultiEnvironmentProject/Database/solid_PPO_dataset.pkl", "rb") as f:
        data = pickle.load(f)

    for i in range(len(data["observations"])):
        for j in range(len(data["observations"][i])):
            data["observations"][i][j] = np.array(data["observations"][i][j])

    for i in range(len(data["actions"])):
        for j in range(len(data["actions"][i])):
            data["actions"][i][j] = np.array(data["actions"][i][j])

    print(data["episodes"])
    print(type(data["observations"]))
    for path in paths:
        with open(path, "rb") as f:
            new_data =pickle.load(f)
        for i in range(len(new_data["observations"])):
            for j in range(len(new_data["observations"][i])):
                new_data["observations"][i][j] = np.array(new_data["observations"][i][j])
        for i in range(len(new_data["actions"])):
            for j in range(len(new_data["actions"][i])):
                new_data["actions"][i][j] = np.array(new_data["actions"][i][j])

        data["observations"] += new_data["observations"]
        data["actions"] += new_data["actions"]
        data["rewards"] += new_data["rewards"]

        


#    with open("MultiEnvironmentProject/Database/solid_PPO_dataset.pkl", "rb") as f:
#        data = pickle.load(f)
#    print(type(data))
#    print(data.keys())
#    print(data["game"])
#    print(data["comulative_reward"])
#    print(data["episodes"])    
#    print(data["steps_per_episode"])
#    print(data["observations"][0][0])
#    #data["observations"] = np.array(data["observations"])
#    
#    for i in range(len(data["observations"])):
#        for j in range(len(data["observations"][i])):
#            data["observations"][i][j] = np.array(data["observations"][i][j])
#    #print(data["observations"][0])
#    for i in range(len(data["actions"])):
#        for j in range(len(data["actions"][i])):
#            data["actions"][i][j] = np.array(data["actions"][i][j])


    
    config = {
        "embedding_dim": 128,
        "max_state_dim" : 500,
        "context_length" : 600,
        "n_layer" : 3,
        "n_head" : 4,
        "max_ep_length" : 600,
        "max_act_dim" : 4,
        "max_dis_act_dims" : [3,3,2,2],
        "img_dim_x" : 1,
        "img_dim_y" : 1,
        "dueling_arch": True
        }
    random.seed(42)
    random.shuffle(data["observations"])
    random.shuffle(data["actions"])
    random.shuffle(data["rewards"])
    print("num of obs:",len(data["observations"]))
    model=DecisionTransformerWithImage(config=config)
    states = data["observations"][:2]
    actions = data["actions"][:2]
    returns_to_go = data["rewards"][:2]
    timesteps = [[x for x in range(600)] for _ in range(len(states))]


    print(model(states=states,
                actions=actions,
                returns_to_go=returns_to_go,
                timesteps=timesteps,
                horizon=1))