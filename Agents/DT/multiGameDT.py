import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import logging
import os
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class GameConfig:
    name: str
    observation_dim: int
    action_dim: int
    action_space: List[int]
    max_episode_len: int
    cumulative_reward: bool

class MultiGameDecisionTransformer(nn.Module):
    def __init__(self, game_configs: Dict[str, GameConfig], hidden_size: int = 128, 
                 n_layer: int = 3, n_head: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.game_configs = game_configs
        self.game_names = list(game_configs.keys())
        self.num_games = len(game_configs)
        self.hidden_size = hidden_size
        
        # Game embedding
        self.game_embedding = nn.Embedding(self.num_games, hidden_size)
        
        # Observation embeddings - one for each game
        self.observation_embeddings = nn.ModuleDict()
        for game_name, config in game_configs.items():
            self.observation_embeddings[game_name] = nn.Sequential(
                nn.Linear(config.observation_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU()
            )
        
        # Action embeddings - one for each game
        self.action_embeddings = nn.ModuleDict()
        for game_name, config in game_configs.items():
            self.action_embeddings[game_name] = nn.Embedding(config.action_dim, hidden_size)
        
        # Return-to-go embedding
        self.return_embedding = nn.Linear(1, hidden_size)
        
        # Timestep embedding
        max_ep_len = max(config.max_episode_len for config in game_configs.values())
        self.timestep_embedding = nn.Embedding(max_ep_len, hidden_size)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_head,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        # Prediction heads - one for each game
        self.state_pred_heads = nn.ModuleDict()
        self.action_pred_heads = nn.ModuleDict()
        
        for game_name, config in game_configs.items():
            self.state_pred_heads[game_name] = nn.Linear(hidden_size, config.observation_dim)
            self.action_pred_heads[game_name] = nn.Linear(hidden_size, config.action_dim)
        
        self.return_pred_head = nn.Linear(hidden_size, 1)
        
    def forward(self, games, states, actions, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        
        # Get game indices
        game_indices = torch.tensor([self.game_names.index(game) for game in games], device=DEVICE)
        game_embs = self.game_embedding(game_indices).unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        # Process states and actions for each game
        state_embs = torch.zeros(batch_size, seq_length, self.hidden_size, device=DEVICE)
        action_embs = torch.zeros(batch_size, seq_length, self.hidden_size, device=DEVICE)
        
        for i, game in enumerate(games):
            obs_dim = self.game_configs[game].observation_dim
            act_dim = self.game_configs[game].action_dim
            # Slice to correct dimension before embedding
            state_embs[i] = self.observation_embeddings[game](states[i, :, :obs_dim])
            # Convert actions to long integers for embedding layer
            action_indices = actions[i].long().squeeze(-1)
            action_embs[i] = self.action_embeddings[game](action_indices)

        # Remove extra dimension if present
        if returns_to_go.dim() == 3 and returns_to_go.shape[-1] == 1:
            returns_to_go = returns_to_go.squeeze(-1)
        if timesteps.dim() == 3 and timesteps.shape[-1] == 1:
            timesteps = timesteps.squeeze(-1)

        # Return-to-go embedding
        return_embs = self.return_embedding(returns_to_go.unsqueeze(-1))  # (batch_size, seq_length, hidden_size)
        
        # Timestep embedding
        timestep_embs = self.timestep_embedding(timesteps)  # (batch_size, seq_length, hidden_size)
        
        # Combine all embeddings
        combined_embs = state_embs + action_embs + return_embs + timestep_embs + game_embs  # All should be (batch_size, seq_length, hidden_size)
        
        # Transformer
        if attention_mask is not None:
            # Create padding mask for transformer
            src_key_padding_mask = (attention_mask == 0)
            output = self.transformer(combined_embs, src_key_padding_mask=src_key_padding_mask)
        else:
            output = self.transformer(combined_embs)
        
        # Predictions - use appropriate heads for each game
        max_obs_dim = max(cfg.observation_dim for cfg in self.game_configs.values())
        max_action_dim = max(cfg.action_dim for cfg in self.game_configs.values())
        state_preds = torch.zeros(batch_size, seq_length, max_obs_dim, device=DEVICE)
        action_preds = torch.zeros(batch_size, seq_length, max_action_dim, device=DEVICE)
        
        for i, game in enumerate(games):
            obs_dim = self.game_configs[game].observation_dim
            act_dim = self.game_configs[game].action_dim
            state_preds[i, :, :obs_dim] = self.state_pred_heads[game](output[i])
            action_preds[i, :, :act_dim] = self.action_pred_heads[game](output[i])
        
        return_preds = self.return_pred_head(output)
        
        return state_preds, action_preds, return_preds

def multi_game_collate_fn(batch):
    """
    Custom collate function that handles different observation and action dimensions
    by padding to the maximum size across all games.
    """
    # Find max dimensions across all games
    max_obs_dim = max(item['states'].shape[1] for item in batch)
    # max_action_dim = max(item['actions'].shape[1] for item in batch)
    
    games = [item['game'] for item in batch]
    
    # Pad states and actions to max dimensions
    padded_states = []
    
    padded_states = []
    
    for item in batch:
        state = item['states']
        state_padding = max_obs_dim - state.shape[1]
        if state_padding > 0:
            state = F.pad(state, (0, state_padding), "constant", 0)
        padded_states.append(state)
    
    states = torch.stack(padded_states)
    actions = torch.stack([item['actions'] for item in batch])
    returns_to_go = torch.stack([item['returns_to_go'] for item in batch])
    timesteps = torch.stack([item['timesteps'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    targets = torch.stack([item['targets'] for item in batch])
    
    return {
        "game": games,
        "states": states,
        "actions": actions,
        "returns_to_go": returns_to_go,
        "timesteps": timesteps,
        "attention_mask": attention_mask,
        "targets": targets
    }

class MultiGameDataset(Dataset):
    def __init__(self, episodes, context_length=20, game_configs=None):
        self.episodes = episodes
        self.context_length = context_length
        self.game_configs = game_configs
        self.sequences = self._create_sequences()
    
    def _create_sequences(self):
        sequences = []
        
        for episode in self.episodes:
            if not episode:
                continue
                
            game_name = episode[0]['game']
            if game_name not in self.game_configs:
                continue
                
            config = self.game_configs[game_name]
            ep_len = len(episode)
            if ep_len < 2:  # Need at least 2 steps for prediction
                continue
                
            # Calculate returns-to-go
            rewards = [step['reward'] for step in episode]
            rtgs = np.zeros_like(rewards, dtype=np.float32)
            cumulative_reward = 0
            for t in reversed(range(ep_len)):
                cumulative_reward += rewards[t]
                rtgs[t] = cumulative_reward
            
            # Create sequences
            for t in range(ep_len - 1):  # Can't predict beyond the last step
                start_idx = max(0, t - self.context_length + 1)
                end_idx = t + 1
                
                # Extract sequence
                seq_states = [step['observation'] for step in episode[start_idx:end_idx]]
                seq_actions = [step['action'] for step in episode[start_idx:end_idx]]
                seq_rtgs = rtgs[start_idx:end_idx]
                seq_timesteps = np.arange(start_idx, end_idx)
                
                # The input for predicting action_t should be action_{t-1}.
                # We prepend a "start" token (0) and remove the last action.
                seq_input_actions = [0] + seq_actions[:-1]
                
                # The target actions are the original, un-shifted actions.
                seq_target_actions = seq_actions
                
                # Padding
                padding_len = self.context_length - len(seq_states)
                
                padded_states = np.concatenate([
                    np.zeros((padding_len, config.observation_dim), dtype=np.float32),
                    np.array(seq_states, dtype=np.float32)
                ], axis=0)
                
                # Pad the new INPUT actions
                padded_input_actions = np.concatenate([
                    np.zeros(padding_len, dtype=np.int64),
                    np.array(seq_input_actions, dtype=np.int64)
                ], axis=0)
                
                padded_rtgs = np.concatenate([
                    np.zeros(padding_len, dtype=np.float32),
                    seq_rtgs
                ], axis=0)
                
                padded_timesteps = np.concatenate([
                    np.zeros(padding_len, dtype=np.int64),
                    seq_timesteps
                ], axis=0)
                
                attention_mask = np.concatenate([
                    np.zeros(padding_len, dtype=np.int64),
                    np.ones(len(seq_states), dtype=np.int64)
                ], axis=0)
                
                # Target is the original sequence of actions, with padding for the loss function.
                padded_targets = np.concatenate([
                    np.full(padding_len, -100, dtype=np.int64),  # -100 is the ignore_index for CrossEntropyLoss
                    np.array(seq_target_actions, dtype=np.int64)
                ], axis=0)
                
                sequences.append({
                    "game": game_name,
                    "states": padded_states,
                    "actions": padded_input_actions, # Use the shifted actions as input
                    "returns_to_go": padded_rtgs.reshape(-1, 1),
                    "timesteps": padded_timesteps.reshape(-1, 1),
                    "attention_mask": attention_mask,
                    "targets": padded_targets
                })
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        item = self.sequences[idx]
        return {
            "game": item["game"],
            "states": torch.tensor(item["states"], dtype=torch.float32),
            "actions": torch.tensor(item["actions"], dtype=torch.int64),
            "returns_to_go": torch.tensor(item["returns_to_go"], dtype=torch.float32),
            "timesteps": torch.tensor(item["timesteps"], dtype=torch.int64),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.int64),
            "targets": torch.tensor(item["targets"], dtype=torch.int64)
        }

def load_multi_game_data(pickle_paths, game_configs):
    all_episodes = []
    
    for pickle_path in pickle_paths:
        logger.info(f"Loading dataset from: {pickle_path}")
        try:
            with open(pickle_path, 'rb') as f:
                dataset = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading {pickle_path}: {e}")
            continue
        
        # Extract game name from path or use default
        game_name = os.path.basename(pickle_path).split('_')[0]
        if game_name not in game_configs:
            logger.warning(f"Game {game_name} not found in config, skipping")
            continue
            
        config = game_configs[game_name]
        
        # Extract data lists
        observations_list = dataset['observations']
        actions_list = dataset['actions']
        rewards_list = dataset['rewards']
        dones_list = dataset['dones']
        
        num_episodes = len(observations_list)
        logger.info(f"Processing {num_episodes} episodes for {game_name}")
        
        for ep_idx in range(num_episodes):
            ep_obs = observations_list[ep_idx]
            ep_act = actions_list[ep_idx]
            ep_rew = rewards_list[ep_idx]
            ep_done = dones_list[ep_idx]
            
            ep_len = len(ep_obs)
            if ep_len == 0:
                continue
                
            episode_data = []
            for t in range(ep_len):
                # Calculate per-step reward
                if config.cumulative_reward:
                    if t == 0:
                        step_reward = float(ep_rew[t])
                    else:
                        step_reward = float(ep_rew[t]) - float(ep_rew[t-1])
                else:
                    step_reward = float(ep_rew[t])
                
                # Discrete action handling:
                # Get the action and flatten it if needed
                action_data = ep_act[t]
                game_name = os.path.basename(pickle_path).split('_')[0]
                if game_name in game_configs:
                    config = game_configs[game_name]
                    if hasattr(config, 'action_space') and config.action_space is not None:
                        # Flatten multi-dimensional action
                        action = flatten_action(action_data, config.action_space)
                    else:
                        # Handle 1D actions
                        if isinstance(action_data, (np.ndarray, list)) and len(action_data) > 1:
                            action = int(np.argmax(action_data))
                        else:
                            action = int(action_data)
                
                episode_data.append({
                    'observation': np.array(ep_obs[t], dtype=np.float32),
                    'action': action,  # Store flattened action
                    'reward': step_reward,
                    'done': bool(ep_done[t]),
                    'game': game_name
                })
            
            all_episodes.append(episode_data)
    
    return all_episodes

def flatten_action(action, action_space):
    """
    Convert a multi-dimensional action to a flattened index.
    Example: [1, 2, 0] with action_space [3, 3, 2] becomes 1*6 + 2*2 + 0 = 10
    """
    if not isinstance(action, (list, np.ndarray)):
        return action  # Already flattened
    
    flat_index = 0
    multiplier = 1
    for i in range(len(action)-1, -1, -1):
        flat_index += action[i] * multiplier
        if i > 0:
            multiplier *= action_space[i]
    return flat_index

def unflatten_action(flat_index, action_space):
    """
    Convert a flattened index back to a multi-dimensional action.
    Example: 10 with action_space [3, 3, 2] becomes [1, 2, 0]
    """
    action = [0] * len(action_space)
    temp_index = flat_index
    
    # Iterate from the first action dimension to the last
    for i in range(len(action_space)):
        # Calculate the product of the sizes of the remaining action dimensions
        prod_next_dims = np.prod(action_space[i+1:]) if i + 1 < len(action_space) else 1
        
        # The action for the current dimension is the integer division
        action[i] = int(temp_index // prod_next_dims)
        
        # Update the index for the next iteration
        temp_index %= prod_next_dims
        
    return action

def train_model(model, optimizer, train_loader, val_loader, num_epochs, model_dir, start_epoch=0):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Try to load previous best_val_loss if it exists
    if start_epoch > 0:
        try:
            with open(os.path.join(model_dir, "training_state.json"), "r") as f:
                state = json.load(f)
                best_val_loss = state.get("best_val_loss", float('inf'))
                logger.info(f"Resuming with best validation loss of {best_val_loss:.4f}")
        except FileNotFoundError:
            logger.warning("Could not find training_state.json. Starting with best_val_loss = inf.")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            games = batch['game']
            states = batch['states'].to(DEVICE)
            actions = batch['actions'].to(DEVICE)
            returns_to_go = batch['returns_to_go'].to(DEVICE)
            timesteps = batch['timesteps'].squeeze(-1).to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)
            
            optimizer.zero_grad()
            
            _, action_preds, _ = model(
                games, states, actions, returns_to_go, timesteps, attention_mask
            )
            
            loss = 0
            
            # Calculate loss per game
            for i, game in enumerate(games):
                game_config = model.game_configs[game]
                
                # Reshape predictions and targets for loss calculation
                # Predictions shape: (Sequence_Length, Action_Dimension)
                # Targets shape: (Sequence_Length)
                game_action_preds = action_preds[i, :, :game_config.action_dim]
                pred_view = game_action_preds.reshape(-1, game_config.action_dim)
                target_view = targets[i].reshape(-1)
                
                # Calculate loss across the entire sequence
                # ignore_index=-100 ensures we don't calculate loss on padded elements
                loss += F.cross_entropy(pred_view, target_view, ignore_index=-100)
            
            loss = loss / len(games)  # Average loss across games in the batch
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                games = batch['game']
                states = batch['states'].to(DEVICE)
                actions = batch['actions'].to(DEVICE)
                returns_to_go = batch['returns_to_go'].to(DEVICE)
                timesteps = batch['timesteps'].squeeze(-1).to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                targets = batch['targets'].to(DEVICE)
                
                _, action_preds, _ = model(
                    games, states, actions, returns_to_go, timesteps, attention_mask
                )
                
                batch_loss = 0
                for i, game in enumerate(games):
                    game_config = model.game_configs[game]
                    
                    # Reshape for loss calculation, same as in training
                    game_action_preds = action_preds[i, :, :game_config.action_dim]
                    pred_view = game_action_preds.reshape(-1, game_config.action_dim)
                    target_view = targets[i].reshape(-1)
                    
                    batch_loss += F.cross_entropy(pred_view, target_view, ignore_index=-100)
                
                batch_loss = batch_loss / len(games)
                val_loss += batch_loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # --- Save Checkpoint ---
        checkpoint_path = os.path.join(model_dir, "latest_checkpoint.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, checkpoint_path)
        logger.info(f"Saved latest checkpoint to {checkpoint_path}")

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(model_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved to {best_model_path} with val loss: {val_loss:.4f}")
            
            # Save the best loss value for resuming
            with open(os.path.join(model_dir, "training_state.json"), "w") as f:
                json.dump({"best_val_loss": best_val_loss}, f)
        
        scheduler.step()
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(model_dir, "loss_plot.png"))
    plt.close()
    
    torch.save(model.state_dict(), os.path.join(model_dir, "final_model.pt"))
    logger.info("Training completed and model saved")

def main():
    # Configuration
    context_length = 20
    hidden_size = 128
    n_layer = 3
    n_head = 4
    dropout = 0.1
    batch_size = 32
    num_epochs = 30
    learning_rate = 1e-4
    
    # Create model directory
    model_dir = "agents\\game_obs\\DT\\MultiGame\\Results\\MultiGame_DT_v1"
    os.makedirs(model_dir, exist_ok=True)
    
    # Options for resuming or using pretrained
    resume_checkpoint = False
    use_pretrained = False
    pretrained_path = "agents\\game_obs\\DT\\MultiGame\\Results\\MultiGame_DT_v1\\best_model.pt"
    
    # Define game configurations
    game_configs = {
        "solid": GameConfig(
            name="solid",
            observation_dim=81,
            action_dim=18,  # 3 × 3 × 2
            action_space=[3, 3, 2],
            max_episode_len=600,
            cumulative_reward=False
        ),
        "pirates": GameConfig(
            name="pirates",
            observation_dim=381,
            action_dim=12,  # 3 × 2 × 2
            action_space=[3, 2, 2], 
            max_episode_len=600,
            cumulative_reward=False
        )
    }
    
    # Define dataset paths
    pickle_paths = [
        "agents\\game_obs\\DT\\Datasets\\pirates\\pirates_PPO_Optimize_Cluster0_Run4_dataset.pkl",
        "agents\\game_obs\\DT\\Datasets\\solid\\solid_PPO_Optimize_Cluster0_Run10_dataset.pkl"
    ]
    
    for path in pickle_paths:
        if not os.path.exists(path):
            logger.error(f"Dataset file not found: {path}. Please ensure the dataset files are available.")
            raise SystemExit(f"Dataset file not found: {path}. Please ensure the dataset files are available.")
    
    # Save game configs
    with open(os.path.join(model_dir, "game_configs.json"), "w") as f:
        config_dict = {name: vars(config) for name, config in game_configs.items()}
        json.dump(config_dict, f, indent=4)
    
    # Load data
    episodes = load_multi_game_data(pickle_paths, game_configs)
    
    # Split into train and validation
    split_idx = int(len(episodes) * 0.9)
    train_episodes = episodes[:split_idx]
    val_episodes = episodes[split_idx:]
    
    # Create datasets
    train_dataset = MultiGameDataset(train_episodes, context_length, game_configs)
    val_dataset = MultiGameDataset(val_episodes, context_length, game_configs)
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=multi_game_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=multi_game_collate_fn
    )
    
    # Create model
    model = MultiGameDecisionTransformer(
        game_configs, hidden_size, n_layer, n_head, dropout
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    start_epoch = 0
    if resume_checkpoint:
        checkpoint_path = os.path.join(model_dir, "latest_checkpoint.pt")
        if os.path.exists(checkpoint_path):
            logger.info(f"Resuming training")
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Loaded model and optimizer. Starting from epoch {start_epoch}")
            train_model(model, optimizer, train_loader, val_loader, num_epochs, model_dir, start_epoch=start_epoch)
        else:
            logger.error(f"Checkpoint file not found: {checkpoint_path}. Cancelling execution.")
            raise SystemExit(f"Checkpoint file not found: {checkpoint_path}. Cancelling execution.")
    elif use_pretrained:
        if os.path.exists(pretrained_path):
            logger.info(f"Loading pretrained model from {pretrained_path}")
            # Create model directory
            new_model_dir = os.path.join(model_dir, "fine_tuned")
            os.makedirs(new_model_dir, exist_ok=True)
            model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))
            train_model(model, optimizer, train_loader, val_loader, num_epochs, new_model_dir)
        else:
            logger.error(f"Pretrained model file not found: {pretrained_path}. Cancelling execution.")
            raise SystemExit(f"Pretrained model file not found: {pretrained_path}. Cancelling execution.")
    else:
        logger.info("Starting training from scratch.")
        train_model(model, optimizer, train_loader, val_loader, num_epochs, model_dir)

if __name__ == "__main__":
    main()