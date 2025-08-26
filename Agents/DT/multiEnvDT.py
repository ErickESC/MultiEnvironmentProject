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


# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. Game Configuration ---

@dataclass
class GameConfig:
    game_id: int
    dataset_path: str
    cumulative_rewards: bool = False
    obs_type: str = "vector" # 'vector', 'image'
    img_size_x: int = None
    img_size_y: int = None
    obs_continuous_dim: int = 0
    obs_discrete_cardinalities: List[int] = field(default_factory=list)
    action_type: str = 'discrete'  # 'discrete', 'continuous', or 'mixed'
    action_continuous_dim: int = 0
    action_discrete_cardinalities: List[int] = field(default_factory=list)

GAME_CONFIGS = {
    #0: GameConfig(
    #    game_id=0,
    #    dataset_path="examples/data/game0_dataset.pkl",
    #    cumulative_rewards=True,
    #    obs_type = 'vector',
    #    img_size_x = None,
    #    img_size_y = None,
    #    obs_continuous_dim=7,
    #    action_type='discrete',
    #    action_discrete_cardinalities=[3, 3, 2]
    #),
    #1: GameConfig(
    #    game_id=1,
    #    dataset_path="examples/data/game1_dataset.pkl",
    #    cumulative_rewards=False,
    #    obs_type = 'vector',
    #    img_size_x = None,
    #    img_size_y = None,
    #    obs_continuous_dim=12,
    #    action_type='continuous',
    #    action_continuous_dim=4
    #),
    #2: GameConfig(
    #    game_id=2,
    #    dataset_path="examples/data/game2_dataset.pkl",
    #    cumulative_rewards=False,
    #    obs_type = 'vector',
    #    img_size_x = None,
    #    img_size_y = None,
    #    obs_continuous_dim=24,
    #    action_type='mixed',
    #    action_continuous_dim=2,
    #    action_discrete_cardinalities=[5, 2]
    #)
    "Pirates": GameConfig(
        game_id=1,
        dataset_path="MultiEnvironmentProject/Database/pirates_PPO_dataset.pkl",
        cumulative_rewards=False,
        obs_type='vector',
        img_size_x=None,
        img_size_y=None,
        obs_continuous_dim=381,
        action_type='discrete',
        action_discrete_cardinalities=[3,2,2]
    )
}

# --- 2. Model Definitions ---
class MultiGameDTConfig(PretrainedConfig):
    model_type = "multigame_dt"

    def __init__(self, game_configs=None, embedding_dim=128, n_layer=3, 
                 n_head=4, context_length=20, max_ep_len=1000, **kwargs):
        self.game_configs = game_configs
        self.embedding_dim = embedding_dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.context_length = context_length
        self.max_ep_len = max_ep_len
        super().__init__(**kwargs)

class MultiGameDecisionTransformer(PreTrainedModel):
    config_class = MultiGameDTConfig

    def __init__(self, config: MultiGameDTConfig):
        super().__init__(config)
        self.embedding_dim = config.embedding_dim
        self.context_length = config.context_length

        # Encoders
        self.state_encoder = nn.Linear(config.game_configs['max_obs_dim'], self.embedding_dim)
        self.action_encoder = nn.Linear(config.game_configs['max_action_dim'], self.embedding_dim)
        self.rtg_encoder = nn.Linear(1, self.embedding_dim)
        self.timestep_encoder = nn.Embedding(config.max_ep_len, self.embedding_dim)
        self.embed_ln = nn.LayerNorm(self.embedding_dim)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=config.n_head,
            dim_feedforward=4 * self.embedding_dim,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layer)
        self.predict_ln = nn.LayerNorm(self.embedding_dim)

        # Prediction heads
        self.action_heads = nn.ModuleDict({
            str(gid): self._create_action_head(gconf)
            for gid, gconf in config.game_configs.items() if isinstance(gconf, GameConfig)
        })

    def _create_action_head(self, game_config: GameConfig):
        head_layers = nn.ModuleDict()
        if game_config.action_type in ['continuous', 'mixed']:
            head_layers['continuous'] = nn.Linear(self.embedding_dim, game_config.action_continuous_dim)
        if game_config.action_type in ['discrete', 'mixed']:
            head_layers['discrete'] = nn.ModuleList([
                nn.Linear(self.embedding_dim, card) for card in game_config.action_discrete_cardinalities
            ])
        return head_layers

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask, game_ids=None):
        batch_size, seq_len = states.shape[0], self.context_length

        # Embed all inputs
        state_embeds = self.state_encoder(states)
        action_embeds = self.action_encoder(actions)
        rtg_embeds = self.rtg_encoder(returns_to_go)
        time_embeds = self.timestep_encoder(timesteps)

        state_embeds += time_embeds
        action_embeds += time_embeds
        rtg_embeds += time_embeds

        # Assemble sequence
        stacked_inputs = torch.stack((rtg_embeds, state_embeds, action_embeds), dim=1)
        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_len, self.embedding_dim)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Create attention mask
        stacked_mask = torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
        stacked_mask = stacked_mask.permute(0, 2, 1).reshape(batch_size, 3 * seq_len)
        inverted_attention_mask = stacked_mask == 0

        # Transformer forward pass
        x = self.transformer(stacked_inputs, src_key_padding_mask=inverted_attention_mask)
        x = self.predict_ln(x)

        # Reshape and get predictions
        x = x.reshape(batch_size, seq_len, 3, self.embedding_dim).permute(0, 2, 1, 3)
        state_preds = x[:,1]  # Predictions from state representations

        # Game-specific predictions
        all_predictions = {}
        for gid, head_module in self.action_heads.items():
            game_preds = {}
            if 'continuous' in head_module:
                game_preds['continuous'] = head_module['continuous'](state_preds)
            if 'discrete' in head_module:
                game_preds['discrete'] = [head(state_preds) for head in head_module['discrete']]
            all_predictions[gid] = game_preds

        return all_predictions


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
        #self.action_encoder = nn.Linear(config.game_configs['max_act_dim'], self.embedding_dim)
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
    def _pad_features(self, input_tensors:torch.Tensor, tensor_type=None) -> tuple[torch.Tensor, torch.Tensor]:
        if tensor_type == "action":
            if self.max_act_dim > input_tensors.size(-1):
                input_tensors = F.pad(input_tensors, (0, 0, 0, self.max_act_dim - input_tensors.size(-1)))

        elif tensor_type =="state":
            if self.max_state_dim > input_tensors.size(-1):
                input_tensors = F.pad(input_tensors, (0, 0, 0, self.max_state_dim - input_tensors.size(-1)))
        else:
            assert(ValueError("tensor_type was not given"))
        return input_tensors

    def _pad_sequences(self, input_tensors:list, tensor_type = None) -> tuple[torch.Tensor, torch.Tensor]:
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
            assert(ValueError("tensor_type was not given"))

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
        States, actions, returns_to_go and timesteps are expected to be a list numpy arrays
        """
        #actions and states are a list of lists of numpy arrays
        #rewards is a list of lists of a list of a single int/float
        batch_size, seq_len = len(states), len(states[0])
        
        #Transform states, actions, rtg, timesteps into a list of torch tensors
        if(type(states) == list):
            print(len(states))
            states = [[torch.from_numpy(single_state) for single_state in state] for state in states]
            states = [torch.stack(state,dim=0) for state in states]
            states = torch.stack(states,dim=0)
            print(states.size())


        if(type(actions) == list):
            #store actions as integers if they are discrete actions, otherwise as floats if they are continous actions
            #currently doesn't accept a mixture of discrete and continous actions in a single game
            actions = [[torch.tensor(self.vocab[tuple(list(action))],dtype=torch.int32) if np.issubdtype(action.dtype, np.integer) 
                        else torch.from_numpy(action.float())
                        for action in seq_actions]
                        for seq_actions in actions]
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

        #Apply padding along the feature dimention and calculate the attention_masks for states, actions, rtg
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

        print(state_embeds.size(), action_embeds.size(),rtg_embeds.size(),time_embeds.size())
        state_embeds += time_embeds
        state_embeds = state_embeds
        rtg_embeds += time_embeds
        rtg_embeds = rtg_embeds
        action_embeds += time_embeds



        # Assemble sequence
        stacked_inputs = torch.stack((rtg_embeds, state_embeds, action_embeds), dim=1)
        print(stacked_inputs.size())
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
        #output = [self.vocab[all_predictions[x].item()] for x in range(batch_size)]
        return decoded
# --- 3. Dataset and Collator ---
class MultiGameDataset(TorchDataset):
    def __init__(self, sequences, context_length, state_mean, state_std):
        self.sequences = sequences
        self.context_length = context_length
        self.state_mean = state_mean
        self.state_std = state_std

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        game_id = seq['game_id']
        
        # Normalize with epsilon to avoid division by zero
        epsilon = 1e-8
        obs_continuous = (seq['obs_continuous'] - self.state_mean[game_id]) / (self.state_std[game_id] + epsilon)
        
        return {
            "game_id": torch.tensor(game_id, dtype=torch.long),
            "states": torch.from_numpy(obs_continuous).float(),
            "actions": torch.from_numpy(seq['actions_one_hot']).float(),
            "returns_to_go": torch.from_numpy(seq['rtgs']).float().unsqueeze(-1),
            "timesteps": torch.from_numpy(seq['timesteps']).long(),
            "attention_mask": torch.from_numpy(seq['mask']).long(),
            "labels_continuous": torch.from_numpy(seq['action_targets_continuous']).float(),
            "labels_discrete": torch.from_numpy(seq['action_targets_discrete']).long(),
        }

@dataclass
class MultiGameDataCollator:
    max_obs_dim: int
    max_action_dim: int
    max_cont_act_dim: int  # Add this parameter
    context_length: int

    def __call__(self, features):
        # Pad continuous labels to max_cont_act_dim
        padded_cont_labels = []
        for f in features:
            cont_label = f["labels_continuous"]
            if len(cont_label.shape) == 1:
                cont_label = cont_label.unsqueeze(0)  # Make it 2D if it's 1D
            # Pad to max_cont_act_dim
            padding = self.max_cont_act_dim - cont_label.shape[-1]
            if padding > 0:
                cont_label = F.pad(cont_label, (0, padding), "constant", 0)
            padded_cont_labels.append(cont_label)

        return {
            "game_id": torch.stack([f["game_id"] for f in features]),
            "states": torch.stack([f["states"] for f in features]),
            "actions": torch.stack([f["actions"] for f in features]),
            "returns_to_go": torch.stack([f["returns_to_go"] for f in features]),
            "timesteps": torch.stack([f["timesteps"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels_continuous": torch.stack(padded_cont_labels),
            "labels_discrete": torch.stack([f["labels_discrete"] for f in features]),
        }


# --- 4. Trainer ---
class MultiGameTrainer(Trainer):
    def __init__(self, *args, game_configs, **kwargs):
        super().__init__(*args, **kwargs)
        self.game_configs = game_configs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        print(inputs.keys())
        print(inputs['actions'][0])
        inputs.pop("num_items_in_batch", None)
        game_ids = inputs.pop("game_id")
        labels_cont = inputs.pop("labels_continuous")
        labels_disc = inputs.pop("labels_discrete")
        attention_mask = inputs['attention_mask']
        all_predictions = model(**inputs)
        total_loss = 0.0

        for gid_int, gconf in self.game_configs.items():
            if not isinstance(gconf, GameConfig): 
                continue

            game_mask = (game_ids == gid_int)
            print("Game mask shape",game_mask.shape)
            print("labels_cont shape",labels_cont.shape)
            print("labels_disc shape",labels_disc.shape)
            if not game_mask.any(): 
                continue

            gid_str = str(gid_int)
            preds = all_predictions[gid_str]
            if 'continuous' in preds:
                print(preds['continuous'].shape)
        
            # Get current game samples
            current_labels_cont = labels_cont[game_mask]  # shape: [batch, seq_len, act_dim]
            current_labels_disc = labels_disc[game_mask]  # shape: [batch, num_disc_actions]
            print("Total shapes",current_labels_disc.shape,current_labels_cont.shape)
            current_mask = attention_mask[game_mask].bool()  # shape: [batch, seq_len]

            # Continuous action loss
            if gconf.action_type in ['continuous', 'mixed'] and 'continuous' in preds:
                cont_preds = preds['continuous'][game_mask]  # shape: [batch, seq_len, act_dim]
                print(cont_preds.shape)
            
                # Flatten predictions and labels while preserving action dimensions
                cont_preds = cont_preds.reshape(-1, cont_preds.shape[-1])  # [batch*seq_len, act_dim]
                print(cont_preds.shape)
                cont_labels = current_labels_cont.reshape(-1, current_labels_cont.shape[-1])  # [batch*seq_len, act_dim]
            
                # Apply mask by repeating it for each action dimension
                print(current_mask.shape)
                print(current_mask.reshape(-1).shape)
                mask_expanded = current_mask.reshape(-1).unsqueeze(1).expand(-1, cont_preds.shape[-1])
                cont_preds = cont_preds[mask_expanded].reshape(-1, cont_preds.shape[-1])
                cont_labels = cont_labels[mask_expanded].reshape(-1, cont_labels.shape[-1])
            
                if cont_preds.shape[0] > 0:  # Only compute loss if we have valid samples
                    total_loss += F.mse_loss(cont_preds, cont_labels)

            # Discrete action loss
            if gconf.action_type in ['discrete', 'mixed'] and 'discrete' in preds:
                for i, head_preds in enumerate(preds['discrete']):
                    print("HEAD PREDS SHAPE", head_preds.shape)
                    disc_preds_i = head_preds[game_mask]  # shape: [batch, seq_len, num_classes]
                    disc_labels_i = current_labels_disc[:, i]  # shape: [batch]
                    # Expand labels to match sequence length and apply mask
                    disc_labels_i = disc_labels_i.unsqueeze(1).expand(-1, disc_preds_i.shape[1])  # [batch, seq_len]
                    print(disc_labels_i.shape)
                    disc_labels_i = disc_labels_i[current_mask]  # [num_valid]
                    disc_preds_i = disc_preds_i[current_mask]  # [num_valid, num_classes]
                    disc_preds_i = torch.argmax(disc_preds_i, dim=1)
                
                    if disc_preds_i.shape[0] > 0:  # Only compute loss if we have valid samples
                        total_loss += F.cross_entropy(disc_preds_i, disc_labels_i)

        return (total_loss, all_predictions) if return_outputs else total_loss


# --- 5. Data Loading ---
def load_and_process_data(game_configs, context_length):



    all_sequences = []
    state_stats = {}
    max_obs_dim = 0
    max_action_dim = 0

    for gid, gconf in game_configs.items():
        if not isinstance(gconf, GameConfig): 
            continue

        with open(gconf.dataset_path, 'rb') as f:
            episodes = pickle.load(f)

        max_obs_dim = max(max_obs_dim, gconf.obs_continuous_dim)
        action_one_hot_size = gconf.action_continuous_dim + sum(gconf.action_discrete_cardinalities)
        max_action_dim = max(max_action_dim, action_one_hot_size)

        # Calculate stats
        all_states = np.concatenate([ep['observations'] for ep in episodes], axis=0)
        state_stats[gid] = {
            'mean': np.mean(all_states, axis=0),
            'std': np.std(all_states, axis=0) + 1e-6
        }

        # Create sequences
        for ep in episodes:
            ep_len = len(ep['observations'])
            for t in range(context_length-1, ep_len):
                start = t - context_length + 1
                
                obs = ep['observations'][start:t+1]
                acts = ep['actions'][start:t+1]
                
                # Create one-hot actions
                acts_one_hot = []
                for a in acts:
                    a_cont = a[:gconf.action_continuous_dim] if gconf.action_type != 'discrete' else []
                    a_disc = a[gconf.action_continuous_dim:] if gconf.action_type != 'continuous' else []
                    one_hot_parts = [np.array(a_cont)]
                    for i, val in enumerate(a_disc):
                        one_hot = np.zeros(gconf.action_discrete_cardinalities[i])
                        one_hot[int(val)] = 1.0
                        one_hot_parts.append(one_hot)
                    acts_one_hot.append(np.concatenate(one_hot_parts))
                
                # Targets
                target = ep['actions'][t]
                target_cont = target[:gconf.action_continuous_dim] if gconf.action_type != 'discrete' else []
                target_disc = target[gconf.action_continuous_dim:] if gconf.action_type != 'continuous' else []
                
                # Ensure continuous targets are 2D [1, action_dim] even for single actions
                target_cont_array = np.array(target_cont)
                if len(target_cont_array.shape) == 1:
                    target_cont_array = target_cont_array.reshape(1, -1)  # Make it 2D
            
                # Ensure discrete targets are 1D array
                target_disc_array = np.array(target_disc, dtype=np.int64)
        
                all_sequences.append({
                    'game_id': gid,
                    'obs_continuous': obs,
                    'actions_one_hot': np.array(acts_one_hot),
                    'rtgs': ep['returns_to_go'][start:t+1],
                    'timesteps': np.arange(start, t+1),
                    'mask': np.ones(context_length),
                    'action_targets_continuous': target_cont_array,  # Now guaranteed 2D
                    'action_targets_discrete': target_disc_array,     # 1D array
                })

    # Pad sequences
    game_configs['max_obs_dim'] = max_obs_dim
    game_configs['max_action_dim'] = max_action_dim
    game_configs['max_cont_act_dim'] = max(
        (gconf.action_continuous_dim for gconf in game_configs.values() 
         if isinstance(gconf, GameConfig)), default=0)
    game_configs['max_disc_act_dim'] = max(
        (len(gconf.action_discrete_cardinalities) for gconf in game_configs.values() 
         if isinstance(gconf, GameConfig)), default=0)

    for seq in all_sequences:
        gconf = game_configs[seq['game_id']]
        # Pad observations
        pad_width = ((0, 0), (0, max_obs_dim - gconf.obs_continuous_dim))
        seq['obs_continuous'] = np.pad(seq['obs_continuous'], pad_width, 'constant')
        # Pad actions
        action_one_hot_size = gconf.action_continuous_dim + sum(gconf.action_discrete_cardinalities)
        pad_width = ((0, 0), (0, max_action_dim - action_one_hot_size))
        seq['actions_one_hot'] = np.pad(seq['actions_one_hot'], pad_width, 'constant')
        # Pad targets
        cont_target = seq['action_targets_continuous']
        if len(cont_target.shape) == 1:
            cont_target = cont_target.reshape(1, -1)  # Ensure 2D
        padding = game_configs['max_cont_act_dim'] - cont_target.shape[-1]
        if padding > 0:
            cont_target = np.pad(cont_target, ((0,0), (0,padding)), 'constant')
        seq['action_targets_continuous'] = cont_target
        seq['action_targets_discrete'] = np.pad(
            seq['action_targets_discrete'], 
            (0, max(0, game_configs['max_disc_act_dim'] - len(seq['action_targets_discrete']))), 
            'constant', constant_values=-100)

    # Split data
    np.random.shuffle(all_sequences)
    split_idx = int(len(all_sequences) * 0.9)
    train_seqs = all_sequences[:split_idx]
    eval_seqs = all_sequences[split_idx:]

    # Pad normalization stats
    state_mean = {}
    state_std = {}
    for gid, stats in state_stats.items():
        gconf = game_configs[gid]
        if gconf.obs_continuous_dim < max_obs_dim:
            pad_width = (0, max_obs_dim - gconf.obs_continuous_dim)
            state_mean[gid] = np.pad(stats['mean'], pad_width, 'constant')
            state_std[gid] = np.pad(stats['std'], pad_width, 'constant')
        else:
            state_mean[gid] = stats['mean']
            state_std[gid] = stats['std']

    train_dataset = MultiGameDataset(train_seqs, context_length, state_mean, state_std)
    eval_dataset = MultiGameDataset(eval_seqs, context_length, state_mean, state_std)

    return train_dataset, eval_dataset, game_configs

# --- 5.2 Dummy data creation ---

def create_dummy_datasets():
    """Creates and verifies dummy data files for demonstration."""
    if not os.path.exists("examples/data"):
        os.makedirs("examples/data")
    
    for gid, gconf in GAME_CONFIGS.items():
        episodes = []
        print(f"\nCreating dummy data for Game {gid}...")
        print(f"Config: obs_dim={gconf.obs_continuous_dim}, "
              f"action_type={gconf.action_type}, "
              f"cont_act_dim={gconf.action_continuous_dim}, "
              f"disc_act_dims={gconf.action_discrete_cardinalities}")
        
        for ep_idx in range(10):  # 10 episodes
            ep_len = np.random.randint(50, 100)
            print(f"\nEpisode {ep_idx} (length {ep_len})")
            
            # Observations
            obs = np.random.randn(ep_len, gconf.obs_continuous_dim)
            print(f"  Observations shape: {obs.shape}")
            
            # Actions
            acts = []
            for t in range(ep_len):
                a_cont = np.random.randn(gconf.action_continuous_dim)
                a_disc = [np.random.randint(0, c) for c in gconf.action_discrete_cardinalities]
                action = np.concatenate([a_cont, a_disc]).astype(np.float32)
                acts.append(action)
                if t == 0:  # Print first action for verification
                    print(f"  First action: cont={a_cont}, disc={a_disc}")
                    print(f"  Full action shape: {action.shape}")
            
            acts = np.array(acts)
            print(f"  All actions shape: {acts.shape}")
            
            # Rewards and RTGs
            rews = np.random.rand(ep_len)
            rtgs = np.array([np.sum(rews[i:]) for i in range(ep_len)])
            
            episodes.append({
                'observations': obs,
                'actions': acts,
                'rewards': rews,
                'returns_to_go': rtgs
            })
            
        # Save the dataset
        with open(gconf.dataset_path, 'wb') as f:
            pickle.dump(episodes, f)
        
        # Verify the saved data
        with open(gconf.dataset_path, 'rb') as f:
            loaded_episodes = pickle.load(f)
            print(f"\nVerification for Game {gid}:")
            print(f"Number of episodes: {len(loaded_episodes)}")
            print(f"First episode observations shape: {loaded_episodes[0]['observations'].shape}")
            print(f"First episode actions shape: {loaded_episodes[0]['actions'].shape}")
            print(f"First episode returns_to_go shape: {loaded_episodes[0]['returns_to_go'].shape}")
    
    logger.info("Dummy datasets created and verified in 'examples/data/'.")
    
def verify_data_consistency(game_configs):
    """Verifies that all datasets match their respective game configurations."""
    for gid, gconf in game_configs.items():
        if not isinstance(gconf, GameConfig):
            continue
            
        print(f"\nVerifying Game {gid}...")
        
        if not os.path.exists(gconf.dataset_path):
            print(f"  Dataset not found at {gconf.dataset_path}")
            continue
            
        with open(gconf.dataset_path, 'rb') as f:
            episodes = pickle.load(f)
            
        for i, ep in enumerate(episodes):
            # Check observations
            print("EPISODES:",ep)
            obs_shape = ep['observations'][0].shape
            if obs_shape[1] != gconf.obs_continuous_dim:
                print(f"  Episode {i}: Observation dimension mismatch! "
                      f"Expected {gconf.obs_continuous_dim}, got {obs_shape[1]}")
            
            # Check actions
            act_shape = ep['actions'].shape
            expected_act_dim = gconf.action_continuous_dim + len(gconf.action_discrete_cardinalities)
            if act_shape[1] != expected_act_dim:
                print(f"  Episode {i}: Action dimension mismatch! "
                      f"Expected {expected_act_dim}, got {act_shape[1]}")
            
            # Check RTGs
            if len(ep['returns_to_go']) != len(ep['observations']):
                print(f"  Episode {i}: RTG length doesn't match observations")
                
        print(f"  Game {gid} verification complete. Found {len(episodes)} episodes.")

def verify_real_data_consistency(game_configs):
    for gid, gconf in game_configs.items():
        if not isinstance(gconf, GameConfig):
            continue
            
        print(f"\nVerifying Game {gid}...")
        
        if not os.path.exists(gconf.dataset_path):
            print(f"  Dataset not found at {gconf.dataset_path}")
            continue
            
        with open(gconf.dataset_path, 'rb') as f:
            episodes = pickle.load(f)
        for i in range(len(episodes)):
            obs_shape = episodes["observations"][0][0].shape
            if obs_shape[0] != gconf.obs_continuous_dim:
                print(f"  Episode {i}: Observation dimension mismatch! "
                      f"Expected {gconf.obs_continuous_dim}, got {obs_shape[1]}")
            act_shape = episodes['actions'][0][0].shape
            print(act_shape)
            expected_act_dim = gconf.action_continuous_dim + len(gconf.action_discrete_cardinalities)
            print(expected_act_dim)
            if act_shape[0] != expected_act_dim:
                print(f"  Episode {i}: Action dimension mismatch! "
                      f"Expected {expected_act_dim}, got {act_shape[1]}")
def load_and_process_real_data(game_configs, context_length):
    all_sequences = []
    state_stats = {}
    max_obs_dim = 0
    max_action_dim = 0

    for gid, gconf in game_configs.items():
        if not isinstance(gconf, GameConfig): 
            continue

        with open(gconf.dataset_path, 'rb') as f:
            episodes = pickle.load(f)

        max_obs_dim = max(max_obs_dim, gconf.obs_continuous_dim)
        action_one_hot_size = gconf.action_continuous_dim + sum(gconf.action_discrete_cardinalities)
        max_action_dim = max(max_action_dim, action_one_hot_size)

        # Calculate stats
        print(episodes.keys())
        all_states = np.concatenate([ep for ep in episodes['observations']], axis=0)
        state_stats[gid] = {
            'mean': np.mean(all_states, axis=0),
            'std': np.std(all_states, axis=0) + 1e-6
        }
        #print(episodes['observations'][0])
        #print(len(episodes['observations'][0]))

        for i in range(len(episodes['observations'])):
            ep_len = len(episodes['observations'][i])
            for t in range(context_length-1, ep_len):
                start = t - context_length + 1    
                obs = episodes['observations'][start:t+1]
                acts = episodes['actions'][start:t+1]

                acts_one_hot = []
                for a in acts:
                    a_cont = a[:gconf.action_continuous_dim] if gconf.action_type != 'discrete' else []
                    a_disc = a[gconf.action_continuous_dim:] if gconf.action_type != 'continuous' else []
                    one_hot_parts = [np.array(a_cont)]
                    for i, val in enumerate(a_disc):
                        one_hot = np.zeros(gconf.action_discrete_cardinalities[i])
                        print(val)
                        one_hot[int(val)] = 1.0
                        one_hot_parts.append(one_hot)
                    acts_one_hot.append(np.concatenate(one_hot_parts))

                target = episodes['actions'][t]
                target_cont = target[:gconf.action_continuous_dim] if gconf.action_type != 'discrete' else []
                target_disc = target[gconf.action_continuous_dim:] if gconf.action_type != 'continuous' else []
                if len(target_cont_array.shape) == 1:
                    target_cont_array = target_cont_array.reshape(1, -1)  # Make it 2D
            
                # Ensure discrete targets are 1D array
                target_disc_array = np.array(target_disc, dtype=np.int64)
        
                all_sequences.append({
                    'game_id': gid,
                    'obs_continuous': obs,
                    'actions_one_hot': np.array(acts_one_hot),
                    'rtgs': episodes['returns_to_go'][start:t+1],
                    'timesteps': np.arange(start, t+1),
                    'mask': np.ones(context_length),
                    'action_targets_continuous': target_cont_array,  # Now guaranteed 2D
                    'action_targets_discrete': target_disc_array,     # 1D array
                })

    # Pad sequences
    game_configs['max_obs_dim'] = max_obs_dim
    game_configs['max_action_dim'] = max_action_dim
    game_configs['max_cont_act_dim'] = max(
        (gconf.action_continuous_dim for gconf in game_configs.values() 
         if isinstance(gconf, GameConfig)), default=0)
    game_configs['max_disc_act_dim'] = max(
        (len(gconf.action_discrete_cardinalities) for gconf in game_configs.values() 
         if isinstance(gconf, GameConfig)), default=0)

    for seq in all_sequences:
        gconf = game_configs[seq['game_id']]
        # Pad observations
        pad_width = ((0, 0), (0, max_obs_dim - gconf.obs_continuous_dim))
        seq['obs_continuous'] = np.pad(seq['obs_continuous'], pad_width, 'constant')
        # Pad actions
        action_one_hot_size = gconf.action_continuous_dim + sum(gconf.action_discrete_cardinalities)
        pad_width = ((0, 0), (0, max_action_dim - action_one_hot_size))
        seq['actions_one_hot'] = np.pad(seq['actions_one_hot'], pad_width, 'constant')
        # Pad targets
        cont_target = seq['action_targets_continuous']
        if len(cont_target.shape) == 1:
            cont_target = cont_target.reshape(1, -1)  # Ensure 2D
        padding = game_configs['max_cont_act_dim'] - cont_target.shape[-1]
        if padding > 0:
            cont_target = np.pad(cont_target, ((0,0), (0,padding)), 'constant')
        seq['action_targets_continuous'] = cont_target
        seq['action_targets_discrete'] = np.pad(
            seq['action_targets_discrete'], 
            (0, max(0, game_configs['max_disc_act_dim'] - len(seq['action_targets_discrete']))), 
            'constant', constant_values=-100)

    # Split data
    np.random.shuffle(all_sequences)
    split_idx = int(len(all_sequences) * 0.9)
    train_seqs = all_sequences[:split_idx]
    eval_seqs = all_sequences[split_idx:]

    # Pad normalization stats
    state_mean = {}
    state_std = {}
    for gid, stats in state_stats.items():
        gconf = game_configs[gid]
        if gconf.obs_continuous_dim < max_obs_dim:
            pad_width = (0, max_obs_dim - gconf.obs_continuous_dim)
            state_mean[gid] = np.pad(stats['mean'], pad_width, 'constant')
            state_std[gid] = np.pad(stats['std'], pad_width, 'constant')
        else:
            state_mean[gid] = stats['mean']
            state_std[gid] = stats['std']

    train_dataset = MultiGameDataset(train_seqs, context_length, state_mean, state_std)
    eval_dataset = MultiGameDataset(eval_seqs, context_length, state_mean, state_std)

    return train_dataset, eval_dataset, game_configs
            

# --- 6. Main Execution ---

if __name__ == "__main__":
    CONTEXT_LENGTH = 20
    with open("MultiEnvironmentProject/Database/pirates_PPO_dataset.pkl", "rb") as f:
        data = pickle.load(f)
    print(data.keys())
    print(data["game"])
    print(data["comulative_reward"])
    print(data["episodes"])    
    print(data["steps_per_episode"])
    print(data["observations"][0][0].shape)
    
    #print(len(data['rewards']))
    #print(len(data['rewards'][0]))
    #print(len(data['rewards'][0][0]))
    #print(len(data["actions"])) # 30 episodes
    #print(len(data["actions"][0])) #600 steps / ep
    #actions and states are a list of lists of numpy arrays
    #rewards is a list of lists of a list of a single int/float
    #print(list(data["actions"]))
    #print(data["rewards"])
    #print(data["dones"])
    # Create dummy data if it doesn't exist
 # if not all(os.path.exists(gconf.dataset_path) for gconf in GAME_CONFIGS.values()):
 #     create_dummy_datasets()
 #     
 # # Verify all datasets
 # print("\nPerforming comprehensive data verification...")
 # verify_real_data_consistency(GAME_CONFIGS)
 # 
 # # 1. Load and process data
 # train_ds, eval_ds, updated_game_configs = load_and_process_real_data(GAME_CONFIGS, CONTEXT_LENGTH)
 # 
 # # 2. Setup Model Configuration
 # model_config = MultiGameDTConfig(
 #     game_configs=updated_game_configs,
 #     context_length=CONTEXT_LENGTH,
 #     embedding_dim=128,
 #     n_layer=4,
 #     n_head=4,
 #     max_ep_len=1000
 # )

 # # 3. Instantiate Model, Collator, and Trainer
 # model = MultiGameDecisionTransformer(model_config)
 # collator = MultiGameDataCollator(
 #     max_obs_dim=updated_game_configs['max_obs_dim'],
 #     max_action_dim=updated_game_configs['max_action_dim'],
 #     max_cont_act_dim=updated_game_configs['max_cont_act_dim'],
 #     context_length=CONTEXT_LENGTH
 # )

 # training_args = TrainingArguments(
 #     output_dir="output/multigame_dt_model",
 #     report_to="none",
 #     num_train_epochs=5,
 #     per_device_train_batch_size=64,
 #     per_device_eval_batch_size=64,
 #     learning_rate=1e-4,
 #     weight_decay=1e-4,
 #     warmup_ratio=0.1,
 #     logging_dir="logs/multigame_dt_logs",
 #     logging_steps=10,
 #     save_strategy="steps",
 #     save_steps=100,
 #     eval_strategy="steps",
 #     eval_steps=100,
 #     remove_unused_columns=False,
 # )

 # trainer = MultiGameTrainer(
 #     model=model,
 #     args=training_args,
 #     game_configs=updated_game_configs,
 #     data_collator=collator,
 #     train_dataset=train_ds,
 #     eval_dataset=eval_ds,
 # )

 # ## 4. Train the model
 # #logger.info("Starting training...")
 # #trainer.train()
 # #logger.info("Training complete!")
    config = {
        "embedding_dim": 128,
        "max_state_dim" : 381,
        "context_length" : 600,
        "horizon" : 2,
        "n_layer" : 3,
        "n_head" : 4,
        "max_ep_length" : 600,
        "max_act_dim" : 3,
        "max_dis_act_dims" : [3,2,2],
        "img_dim_x" : 1,
        "img_dim_y" : 1,
        "dueling_arch": True
        }
    model=DecisionTransformerWithImage(config=config)
    states = data["observations"][:5]
    actions = data["actions"][:5]
    returns_to_go = data["rewards"][:5]
    timesteps = [[x for x in range(600)] for _ in range(5)]


    print(model(states=states,
                actions=actions,
                returns_to_go=returns_to_go,
                timesteps=timesteps,
                horizon=1))