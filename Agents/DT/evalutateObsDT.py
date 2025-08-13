import torch
from transformers import (
    DecisionTransformerConfig,
    DecisionTransformerModel,
)
import numpy as np
import random
import logging
import os
import json

# import imageio

from affectively.environments.solid_game_obs import SolidEnvironmentGameObs
from affectively.environments.base import compute_confidence_interval
from trainObsDT import plot_metrics

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Evaluation (Online Rollout) ---
def evaluate_online(env_creator, model_path, target_rtg, num_episodes=10):
    """Evaluates the trained model online in the environment."""

    logger.info(f"Starting online evaluation with target RTG: {target_rtg}")
    if not model_path or not os.path.exists(model_path):
         logger.error(f"Model path not found: {model_path}")
         return None, None # Cannot evaluate without model

    # Load model and normalization stats
    try:
        model_config = DecisionTransformerConfig.from_pretrained(model_path)
        model = DecisionTransformerModel.from_pretrained(model_path)
        model.to(DEVICE).eval()

        stats_path = os.path.join(model_path, "normalization_stats.npz")
        stats = np.load(stats_path)
        state_mean = stats['mean']
        state_std = stats['std']
        max_ep_len = int(stats['max_ep_len'])
    except Exception as e:
         logger.error(f"Error loading model or stats from {model_path}: {e}")
         return None, None


    # Action space details (must match training)
    _num_actions_per_dim = NUM_ACTIONS_PER_DIM
    _num_action_dims = len(_num_actions_per_dim)
    _action_slice_starts = np.concatenate(([0], np.cumsum(_num_actions_per_dim)[:-1]))
    _concatenated_action_dim = sum(_num_actions_per_dim)

    # Create environment
    try:
        env = env_creator()
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        return None, None


    episode_returns = []
    episode_lengths = []
    arousal, scores = [], []

    for ep in range(num_episodes):
        try:
            state = env.reset()
            state = np.array(state, dtype=np.float32) # Ensure numpy float32
        except Exception as e:
            logger.error(f"Environment reset failed: {e}")
            continue # Skip episode if reset fails

        done = False
        ep_return = 0
        ep_len = 0
        current_target_rtg = float(target_rtg) # Ensure target is float

        # Initialize context buffers
        # States need normalization
        norm_state = (state - state_mean) / state_std
        context_states = [np.zeros_like(state, dtype=np.float32)] * (CONTEXT_LENGTH - 1) + [norm_state]
        # Actions are one-hot encoded for the model, but store indices for history
        context_actions_one_hot = [np.zeros(_concatenated_action_dim, dtype=np.float32)] * CONTEXT_LENGTH
        context_rtgs = [np.array([current_target_rtg], dtype=np.float32)] * CONTEXT_LENGTH
        context_timesteps = [np.array([0], dtype=np.int64)] * CONTEXT_LENGTH


        while not done:
            # Prepare model input tensors from context
            states_tensor = torch.tensor(np.array(context_states)[-CONTEXT_LENGTH:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            actions_tensor = torch.tensor(np.array(context_actions_one_hot)[-CONTEXT_LENGTH:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            rtgs_tensor = torch.tensor(np.array(context_rtgs)[-CONTEXT_LENGTH:], dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Extract scalar timesteps into a list
            current_context_timesteps_list = [ts[0] for ts in context_timesteps[-CONTEXT_LENGTH:]]
            # Create a 1D numpy array
            timesteps_np = np.array(current_context_timesteps_list, dtype=np.int64) # Shape (K,)
            # Convert to tensor and add batch dimension
            timesteps_tensor = torch.tensor(timesteps_np, dtype=torch.long).unsqueeze(0).to(DEVICE) # Shape (1, K)

            attn_mask_tensor = torch.ones_like(timesteps_tensor).to(DEVICE) # Shape (1, K)

            with torch.no_grad():
                outputs = model(
                    states=states_tensor,
                    actions=actions_tensor,
                    returns_to_go=rtgs_tensor,
                    timesteps=timesteps_tensor, # Pass the (1, K) tensor
                    attention_mask=attn_mask_tensor,
                    return_dict=True,
                )
                # Get logits for the last timestep prediction
                logits = outputs.action_preds[0, -1] # Shape (concatenated_logit_dim,) e.g., (8,)

            # Sample action INDICES from the logits for each dimension
            predicted_action_indices = np.zeros(_num_action_dims, dtype=np.int64)
            predicted_action_one_hot = np.zeros(_concatenated_action_dim, dtype=np.float32) # For next input
            for i in range(_num_action_dims):
                start_idx = _action_slice_starts[i]
                end_idx = start_idx + _num_actions_per_dim[i]
                dim_logits = logits[start_idx:end_idx]
                # Take argmax for deterministic evaluation
                action_idx = torch.argmax(dim_logits).item()
                predicted_action_indices[i] = action_idx
                # Set the corresponding one-hot bit for the next input
                predicted_action_one_hot[start_idx + action_idx] = 1.0

            # Step environment with action INDICES
            try:
                 next_state, reward, terminated, _ = env.step(predicted_action_indices)
                 next_state = np.array(next_state, dtype=np.float32) # Ensure numpy float32
                 reward = float(reward) # Ensure float
                 done = terminated
            except Exception as e:
                 logger.error(f"Environment step failed: {e}")
                 done = True # End episode if step fails


            # Update context buffers for the next iteration
            norm_next_state = (next_state - state_mean) / state_std
            context_states.append(norm_next_state)
            context_actions_one_hot.append(predicted_action_one_hot) # Append the one-hot action taken
            current_target_rtg -= reward
            context_rtgs.append(np.array([current_target_rtg], dtype=np.float32))
            current_timestep = context_timesteps[-1][0] + 1
            # Clamp timestep to avoid exceeding max_ep_len used for embeddings
            context_timesteps.append(np.array([min(current_timestep, max_ep_len - 1)], dtype=np.int64))

            # Remove oldest entries from context buffers
            context_states.pop(0)
            context_actions_one_hot.pop(0)
            context_rtgs.pop(0)
            context_timesteps.pop(0)

            ep_return += reward
            ep_len += 1
            state = next_state # Update state for next loop

            if ep_len >= max_ep_len: # Add safety break
                 logger.warning(f"Episode {ep+1} reached max length {max_ep_len}, terminating.")
                 done = True


        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        arousal.append(np.mean(env.arousal_trace))
        scores.append(env.best_score)
        env.best_score = 0
        env.arousal_trace.clear()
        logger.info(f"Episode {ep+1}/{num_episodes}: Return={ep_return:.2f}, Length={ep_len}, Arousal={arousal[-1]:.2f}, Score={scores[-1]:.2f}")

    if not episode_returns:
        logger.warning("No episodes completed successfully during evaluation.")
        return None, None

    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    mean_length = np.mean(episode_lengths)
    logger.info("-" * 30)
    logger.info(f"Evaluation Results (Target RTG: {target_rtg}):")
    logger.info(f"Mean Return: {mean_return:.2f} +/- {std_return:.2f}")
    logger.info(f"Mean Length: {mean_length:.2f}")
    logger.info("-" * 30)
    return mean_return, std_return, arousal, scores

def create_dummy_env():
        logger.warning("Using dummy environment for testing. This is not a real environment.")
        class DummyEnv:
            def __init__(self):
                self.state_shape = (STATE_DIM,)
                self.action_space_dims = NUM_ACTION_DIMS
                self.observation_space = type('space', (), {'shape': self.state_shape})()
                self.action_space = type('space', (), {'shape': (self.action_space_dims,)})()
                self._step_count = 0
                self._max_steps = 200 # Limit dummy episode length

            def reset(self, seed=None):
                self._step_count = 0
                state = np.random.rand(*self.state_shape).astype(np.float32)
                return state, {} # Return state and empty info dict

            def step(self, action):
                # print(f"Step: {self._step_count}, Action: {action}") # Debug print
                if not isinstance(action, np.ndarray) or action.shape != (self.action_space_dims,):
                     # Simple check
                     print(f"Warning: Received invalid action shape: {action.shape if hasattr(action, 'shape') else type(action)}")

                self._step_count += 1
                next_state = np.random.rand(*self.state_shape).astype(np.float32)
                reward = random.uniform(0.05, 0.15) # Simulate rewards
                done = False
                if self._step_count >= self._max_steps:
                    done = True
                info = {} # Return empty info dict
                return next_state, reward, done, info
        return DummyEnv()
    
# To record a GIF of an episode, we need to run the model in the environment and capture frames.

def record_gif_episode(env_creator, model_path, target_rtg, gif_path="episode.gif", max_frames=200):
    """Runs one episode, records frames, and saves a GIF."""
    # Load model and normalization stats (reuse your logic)
    model_config = DecisionTransformerConfig.from_pretrained(model_path)
    model = DecisionTransformerModel.from_pretrained(model_path)
    model.to(DEVICE).eval()
    stats_path = os.path.join(model_path, "normalization_stats.npz")
    stats = np.load(stats_path)
    state_mean = stats['mean']
    state_std = stats['std']
    max_ep_len = int(stats['max_ep_len'])

    _num_actions_per_dim = NUM_ACTIONS_PER_DIM
    _num_action_dims = len(_num_actions_per_dim)
    _action_slice_starts = np.concatenate(([0], np.cumsum(_num_actions_per_dim)[:-1]))
    _concatenated_action_dim = sum(_num_actions_per_dim)

    env = env_creator()
    state = env.reset()
    if isinstance(state, tuple):  # Some envs return (obs, info)
        state = state[0]
    state = np.array(state, dtype=np.float32)
    norm_state = (state - state_mean) / state_std
    context_states = [np.zeros_like(state, dtype=np.float32)] * (CONTEXT_LENGTH - 1) + [norm_state]
    context_actions_one_hot = [np.zeros(_concatenated_action_dim, dtype=np.float32)] * CONTEXT_LENGTH
    context_rtgs = [np.array([target_rtg], dtype=np.float32)] * CONTEXT_LENGTH
    context_timesteps = [np.array([0], dtype=np.int64)] * CONTEXT_LENGTH

    frames = []
    done = False
    ep_len = 0
    current_target_rtg = float(target_rtg)

    # --- Capture initial frame ---
    if hasattr(env, "render"):
        frame = env.render(mode="rgb_array") if "mode" in env.render.__code__.co_varnames else env.render()
        if frame is not None:
            frames.append(frame)
        else:
            print("Warning: Initial render returned None.")
    else:
        print("Warning: Environment does not support render.")
    
    while not done and ep_len < max_frames:
        states_tensor = torch.tensor(np.array(context_states)[-CONTEXT_LENGTH:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        actions_tensor = torch.tensor(np.array(context_actions_one_hot)[-CONTEXT_LENGTH:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        rtgs_tensor = torch.tensor(np.array(context_rtgs)[-CONTEXT_LENGTH:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        current_context_timesteps_list = [ts[0] for ts in context_timesteps[-CONTEXT_LENGTH:]]
        timesteps_np = np.array(current_context_timesteps_list, dtype=np.int64)
        timesteps_tensor = torch.tensor(timesteps_np, dtype=torch.long).unsqueeze(0).to(DEVICE)
        attn_mask_tensor = torch.ones_like(timesteps_tensor).to(DEVICE)

        with torch.no_grad():
            outputs = model(
                states=states_tensor,
                actions=actions_tensor,
                returns_to_go=rtgs_tensor,
                timesteps=timesteps_tensor,
                attention_mask=attn_mask_tensor,
                return_dict=True,
            )
            logits = outputs.action_preds[0, -1]

        predicted_action_indices = np.zeros(_num_action_dims, dtype=np.int64)
        predicted_action_one_hot = np.zeros(_concatenated_action_dim, dtype=np.float32)
        for i in range(_num_action_dims):
            start_idx = _action_slice_starts[i]
            end_idx = start_idx + _num_actions_per_dim[i]
            dim_logits = logits[start_idx:end_idx]
            action_idx = torch.argmax(dim_logits).item()
            predicted_action_indices[i] = action_idx
            predicted_action_one_hot[start_idx + action_idx] = 1.0

        next_state, reward, terminated, _ = env.step(predicted_action_indices)
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        next_state = np.array(next_state, dtype=np.float32)
        done = terminated

        norm_next_state = (next_state - state_mean) / state_std
        context_states.append(norm_next_state)
        context_actions_one_hot.append(predicted_action_one_hot)
        current_target_rtg -= reward
        context_rtgs.append(np.array([current_target_rtg], dtype=np.float32))
        current_timestep = context_timesteps[-1][0] + 1
        context_timesteps.append(np.array([min(current_timestep, max_ep_len - 1)], dtype=np.int64))
        context_states.pop(0)
        context_actions_one_hot.pop(0)
        context_rtgs.pop(0)
        context_timesteps.pop(0)

        ep_len += 1

        # --- Capture frame ---
        if hasattr(env, "render"):
            frame = env.render(mode="rgb_array") if "mode" in env.render.__code__.co_varnames else env.render()
            if frame is not None:
                frames.append(frame)
            else:
                print(f"Warning: Render returned None at frame {ep_len}.")
        else:
            print("Warning: Environment does not support render.")
        
    # Save GIF if any frames were captured
    if frames:
        imageio.mimsave(gif_path, frames, fps=15)
        print(f"Saved episode GIF to {gif_path}")
    else:
        print("No valid frames were captured. GIF was not saved.")


NUM_ACTIONS_PER_DIM = [3, 3, 2]  # <<< --- MUST BE SET CORRECTLY
NUM_ACTION_DIMS = len(NUM_ACTIONS_PER_DIM)
STATE_DIM = 81
CONTEXT_LENGTH = 5  # K: How many steps the model sees
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Main Execution ---
if __name__ == "__main__":

    # reward_type = 'arousal'
    # reward_type = 'score'
    reward_type = 'blended'
    
    data_from = 'PPO'
    # data_from = 'Explore'
    
    # Target return for evaluation
    target_return = 7000
    # target_return = 17
    
    # name = f"{data_from}_{label}_{reward_type}_SolidObs_DT"
    # name = f"ODT_Optimize_14k_v300"
    # name = f"ODT_Blended_14k_v300"
    # name = f"ODT_Arousal_14k_v300"
    name = "testingDT_on_newAF_final"
    
    weight = 0.5

    if weight == 0:
        label = 'Optimize'
    elif weight == 0.5:
        label = 'Blended'
    else:
        label = 'Arousal'
    
    discretize=0
    
    if data_from == 'Explore':
        discretize=1
    
    # Set the path to the saved model artifacts
    final_model_path = f"agents\\game_obs\\DT\\Results\\{name}"
    # final_model_path = f"examples\\Agents\\DT\\Results\\Explore_Blended_moreTrained_DT"
    # final_model_path = f"examples\\Agents\\DT\\Results\\preTrained\\PPO_Optimize_score_SolidObs_DT_final"
    
    print(f'Starting to evaluate {name}')
    
    def create_env():
        env = SolidEnvironmentGameObs(
                    id_number=0,
                    weight=0,
                    graphics=True,
                    cluster=0,
                    target_arousal=0,
                    period_ra=0,
                    discretize=0
                )
        sideChannel = env.customSideChannel
        env.targetSignal = np.ones
        return env

    # Run evaluation
    mean_return, std_return, arousal, scores = evaluate_online(
        env_creator=create_env,
        model_path=final_model_path,
        target_rtg=target_return,
        num_episodes=10
    )
    
    print(f"Best Score: {compute_confidence_interval(scores)}, Mean Arousal: {compute_confidence_interval(arousal)}")
    print(f'Done evaluating {name} in {label}')
    
    # --- Record a GIF of one episode ---
    # record_gif_episode(
    #     env_creator=create_env,
    #     model_path=final_model_path,
    #     target_rtg=target_return,
    #     gif_path=f"examples\\Agents\\DT\\GIF\{data_from}_{label}.gif",
    #     max_frames=200
    # )