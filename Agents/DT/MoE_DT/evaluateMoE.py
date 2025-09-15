import torch
import torch.nn as nn
import numpy as np
import os
import json
from typing import Dict, List, Any, Tuple
import logging
from trainMoE import MultiGameDecisionTransformer, GameConfig, DEVICE, unflatten_action

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiGameDTEvaluator:
    def __init__(self, model_path, game_configs_path):
        # Load game configs
        with open(game_configs_path, "r") as f:
            config_dict = json.load(f)
        
        self.game_configs = {}
        for name, config in config_dict.items():
            # Ensure action_space is loaded from config
            self.game_configs[name] = GameConfig(**config)
        
        # Create model with the same parameters as training
        hidden_size = 128
        n_layer = 3
        n_head = 4
        dropout = 0.1
        
        self.model = MultiGameDecisionTransformer(
            self.game_configs, hidden_size, n_layer, n_head, dropout, experts=True, num_of_features=10
        ).to(DEVICE)
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        
        logger.info(f"Loaded model from {model_path}")
    
    def evaluate_online(self, env, game_name, target_rtg, num_episodes=10):
        if game_name not in self.game_configs:
            raise ValueError(f"Game {game_name} not found in config")
        
        config = self.game_configs[game_name]
        context_length = 20  # Should match training
        
        episode_returns = []
        
        for ep in range(num_episodes):
            state = env.reset()
            if hasattr(state, '__len__') and len(state) == 2:  # Handle (obs, info) format
                state = state[0]
            state = np.array(state, dtype=np.float32)
            if state.shape != (config.observation_dim,):
                logger.warning(f"Initial state shape mismatch: got {state.shape}, expected {(config.observation_dim,)}. Padding or truncating.")
                if state.size < config.observation_dim:
                    padded = np.zeros(config.observation_dim, dtype=np.float32)
                    padded[:state.size] = state.flatten()
                    state = padded
                else:
                    state = state.flatten()[:config.observation_dim]
            
            done = False
            ep_return = 0
            ep_len = 0
            
            # Initialize context buffers
            context_states = [np.zeros(config.observation_dim, dtype=np.float32) for _ in range(context_length - 1)] + [state]
            context_actions = [0] * context_length  # Use integer indices for flattened actions
            context_rtgs = [np.array([target_rtg], dtype=np.float32)] * context_length
            context_timesteps = [np.array([0], dtype=np.int64)] * context_length
            
            while not done:
                # Prepare inputs
                states_np = np.stack([np.asarray(s, dtype=np.float32) for s in context_states[-context_length:]])
                states_tensor = torch.tensor(states_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                actions_np = np.array(context_actions[-context_length:], dtype=np.int64)
                actions_tensor = torch.tensor(actions_np, dtype=torch.int64).unsqueeze(0).to(DEVICE)
                rtgs_tensor = torch.tensor(np.array(context_rtgs)[-context_length:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                # Extract scalar timesteps
                current_context_timesteps_list = [ts[0] for ts in context_timesteps[-context_length:]]
                timesteps_np = np.array(current_context_timesteps_list, dtype=np.int64)
                timesteps_tensor = torch.tensor(timesteps_np, dtype=torch.int64).unsqueeze(0).to(DEVICE)
                
                attn_mask_tensor = torch.ones_like(timesteps_tensor).to(DEVICE)
                
                # Get action prediction
                with torch.no_grad():
                    _, action_preds, _ = self.model(
                        [game_name], states_tensor, actions_tensor, rtgs_tensor, 
                        timesteps_tensor, attn_mask_tensor
                    )
                logger.info(f"action_preds: {action_preds}")
                # For discrete actions, take the argmax of the logits
                predicted_action_logits = action_preds[0, -1].cpu().numpy()
                predicted_flattened_action = np.argmax(predicted_action_logits)  # Get the flattened action index
                
                logger.info(f"predicted_action_logits: {predicted_action_logits}")
                logger.info(f"predicted_flattened_action: {predicted_flattened_action}")
                
                # Convert flattened action back to multi-dimensional action for the environment
                if hasattr(config, 'action_space') and config.action_space is not None:
                    predicted_action = unflatten_action(predicted_flattened_action, config.action_space)
                else:
                    predicted_action = predicted_flattened_action
                
                logger.info(f"predicted_action: {predicted_action}")
                
                # Step environment
                next_state, reward, done, _ = env.step(predicted_action)
                
                # Update context buffers
                next_state = np.asarray(next_state, dtype=np.float32)
                if next_state.shape != (config.observation_dim,):
                    logger.warning(f"State shape mismatch: got {next_state.shape}, expected {(config.observation_dim,)}. Padding or truncating.")
                    if next_state.size < config.observation_dim:
                        # Pad with zeros
                        padded = np.zeros(config.observation_dim, dtype=np.float32)
                        padded[:next_state.size] = next_state.flatten()
                        next_state = padded
                    else:
                        # Truncate
                        next_state = next_state.flatten()[:config.observation_dim]
                context_states.append(next_state)
                context_actions.append(predicted_flattened_action)
                target_rtg -= reward
                context_rtgs.append(np.array([target_rtg], dtype=np.float32))
                current_timestep = context_timesteps[-1][0] + 1
                context_timesteps.append(np.array([min(current_timestep, config.max_episode_len - 1)], dtype=np.int64))
                
                # Remove oldest entries
                context_states.pop(0)
                context_actions.pop(0)
                context_rtgs.pop(0)
                context_timesteps.pop(0)
                
                ep_return += reward
                ep_len += 1
                
                if ep_len >= config.max_episode_len:
                    logger.warning(f"Episode reached max length {config.max_episode_len}, terminating.")
                    done = True
            
            episode_returns.append(ep_return)
            logger.info(f"Episode {ep+1}/{num_episodes}: Return={ep_return:.2f}, Length={ep_len}")
        
        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        
        logger.info(f"Evaluation Results for {game_name}:")
        logger.info(f"Mean Return: {mean_return:.2f} +/- {std_return:.2f}")
        
        return mean_return, std_return, episode_returns

def main():
    # Configuration
    model_path = "agents\\game_obs\\DT\\MultiGame\\Results\\MoE\\best_model_MoE.pt"
    game_configs_path = "agents\\game_obs\\DT\\MultiGame\\Results\\MultiGame_DT_v1\\game_configs.json"
    
    # Initialize evaluator
    evaluator = MultiGameDTEvaluator(model_path, game_configs_path)
    
    # Example usage for Solid game
    # try:
    #     from affectively.environments.solid_game_obs import SolidEnvironmentGameObs
        
    #     env = SolidEnvironmentGameObs(
    #         id_number=0,
    #         weight=0,
    #         graphics=True,
    #         cluster=0,
    #         target_arousal=0,
    #         period_ra=0,
    #         discretize=0
    #     )
        
    #     mean_return, std_return, returns = evaluator.evaluate_online(
    #         env, "solid", target_rtg=16.5, num_episodes=5
    #     )
        
    #     logger.info(f"Solid Game - Mean Return: {mean_return:.2f}, Std: {std_return:.2f}")
        
    # except ImportError:
    #     logger.warning("Solid environment not available, skipping evaluation")
    
    # Example usage for Pirates game
    try:
        from affectively.environments.pirates_game_obs import PiratesEnvironmentGameObs
        
        env = PiratesEnvironmentGameObs(
            id_number=0,
            weight=0,
            graphics=True,
            cluster=0,
            target_arousal=0,
            period_ra=0,
            discretize=0
        )
        
        mean_return, std_return, returns = evaluator.evaluate_online(
            env, "pirates", target_rtg=16.5, num_episodes=5
        )
        
        logger.info(f"Pirates Game - Mean Return: {mean_return:.2f}, Std: {std_return:.2f}")
        
    except ImportError:
        logger.warning("Pirates environment not available, skipping evaluation")

if __name__ == "__main__":
    main()