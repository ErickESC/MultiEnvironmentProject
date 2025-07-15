import torch
from transformers import (
    DecisionTransformerConfig,
    DecisionTransformerModel,
)
import numpy as np
import random
import logging
import os
import argparse # Added for command-line arguments

from affectively_environments.envs.solid_game_obs import SolidEnvironmentGameObs
ENV_AVAILABLE = True

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# These need to match the training configuration of the model being loaded
NUM_ACTIONS_PER_DIM = [3, 3, 2]
NUM_ACTION_DIMS = len(NUM_ACTIONS_PER_DIM)
STATE_DIM = 54 # <<<--- Ensure this matches the trained model's state_dim (54 or 7?)
CONTEXT_LENGTH = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Evaluation Function ---
def evaluate_online(env_creator, model_path, target_rtg, num_episodes=10):
    """Evaluates the trained model online in the environment."""

    logger.info(f"Starting online evaluation with target RTG: {target_rtg}")
    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Using Device: {DEVICE}")

    if not model_path or not os.path.exists(model_path):
         logger.error(f"Model path not found: {model_path}")
         return None, None

    # Load model and normalization stats
    try:
        model_config = DecisionTransformerConfig.from_pretrained(model_path)
        # --- STATE DIMENSION CHECK ---
        if model_config.state_dim != STATE_DIM:
            logger.error(f"State dimension mismatch! Model trained with {model_config.state_dim}, but script configured for {STATE_DIM}. Check STATE_DIM constant.")
            return None, None
        # --- END CHECK ---

        model = DecisionTransformerModel.from_pretrained(model_path)
        model.to(DEVICE).eval()

        stats_path = os.path.join(model_path, "normalization_stats.npz")
        if not os.path.exists(stats_path):
             logger.error(f"Normalization stats file not found: {stats_path}")
             return None, None
        stats = np.load(stats_path)
        state_mean = stats['mean']
        state_std = stats['std']
        max_ep_len = int(stats['max_ep_len'])

        # --- STATS DIMENSION CHECK ---
        if state_mean.shape[0] != STATE_DIM or state_std.shape[0] != STATE_DIM:
             logger.error(f"Normalization stats dimension mismatch! Expected {STATE_DIM}, got mean={state_mean.shape}, std={state_std.shape}. Model might be trained with different STATE_DIM.")
             return None, None
        # --- END CHECK ---

    except Exception as e:
         logger.error(f"Error loading model or stats from {model_path}: {e}", exc_info=True)
         return None, None

    # Action space details (derived from constants)
    _num_actions_per_dim = NUM_ACTIONS_PER_DIM
    _num_action_dims = len(_num_actions_per_dim)
    _action_slice_starts = np.concatenate(([0], np.cumsum(_num_actions_per_dim)[:-1]))
    _concatenated_action_dim = sum(_num_actions_per_dim) # 8

    # Create environment
    try:
        env = env_creator()
        logger.info(f"Successfully created environment: {type(env)}")
    except Exception as e:
        logger.error(f"Failed to create environment: {e}", exc_info=True)
        return None, None

    episode_returns = []
    episode_lengths = []

    for ep in range(num_episodes):
        logger.info(f"--- Starting Evaluation Episode {ep+1}/{num_episodes} ---")
        try:
            # Assuming env.reset() returns only state if no info dict provided
            reset_output = env.reset()
            if isinstance(reset_output, tuple) and len(reset_output) == 2:
                state, _ = reset_output # Handle gym-like tuple return
            else:
                state = reset_output # Assume only state is returned

            state = np.array(state, dtype=np.float32)
            if state.shape != (STATE_DIM,):
                logger.error(f"Initial state shape mismatch! Expected ({STATE_DIM},), got {state.shape}. Check env.reset() and STATE_DIM.")
                continue # Skip episode
        except Exception as e:
            logger.error(f"Environment reset failed for episode {ep+1}: {e}", exc_info=True)
            continue # Skip episode if reset fails

        done = False
        ep_return = 0.0
        ep_len = 0
        current_target_rtg = float(target_rtg)

        # Initialize context buffers
        try:
            norm_state = (state - state_mean) / state_std
            context_states = [np.zeros_like(state, dtype=np.float32)] * (CONTEXT_LENGTH - 1) + [norm_state]
            context_actions_one_hot = [np.zeros(_concatenated_action_dim, dtype=np.float32)] * CONTEXT_LENGTH
            context_rtgs = [np.array([current_target_rtg], dtype=np.float32)] * CONTEXT_LENGTH
            # Ensure initial timesteps are arrays within the list
            context_timesteps = [np.array([0], dtype=np.int64)] * CONTEXT_LENGTH
        except Exception as e:
             logger.error(f"Error initializing context buffers for episode {ep+1}: {e}. Check state dimensions.", exc_info=True)
             continue # Skip episode

        while not done:
            # Prepare model input tensors from context
            try:
                states_tensor = torch.tensor(np.array(context_states)[-CONTEXT_LENGTH:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                actions_tensor = torch.tensor(np.array(context_actions_one_hot)[-CONTEXT_LENGTH:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                rtgs_tensor = torch.tensor(np.array(context_rtgs)[-CONTEXT_LENGTH:], dtype=torch.float32).unsqueeze(0).to(DEVICE)

                current_context_timesteps_list = [ts.item() for ts in context_timesteps[-CONTEXT_LENGTH:]] # Get scalar values
                timesteps_np = np.array(current_context_timesteps_list, dtype=np.int64)
                timesteps_tensor = torch.tensor(timesteps_np, dtype=torch.long).unsqueeze(0).to(DEVICE) # Shape (1, K)

                attn_mask_tensor = torch.ones_like(timesteps_tensor).to(DEVICE)
            except Exception as e:
                 logger.error(f"Error creating tensors for model input at step {ep_len}: {e}", exc_info=True)
                 break # End this episode

            # Get action from model
            try:
                with torch.no_grad():
                    outputs = model(
                        states=states_tensor, actions=actions_tensor,
                        returns_to_go=rtgs_tensor, timesteps=timesteps_tensor,
                        attention_mask=attn_mask_tensor, return_dict=True,
                    )
                    logits = outputs.action_preds[0, -1] # Shape (concatenated_logit_dim,)
            except Exception as e:
                 logger.error(f"Error during model forward pass at step {ep_len}: {e}", exc_info=True)
                 break # End this episode

            # Decode action from logits
            predicted_action_indices = np.zeros(_num_action_dims, dtype=np.int64)
            predicted_action_one_hot = np.zeros(_concatenated_action_dim, dtype=np.float32)
            for i in range(_num_action_dims):
                start_idx = _action_slice_starts[i]
                end_idx = start_idx + _num_actions_per_dim[i]
                dim_logits = logits[start_idx:end_idx]
                action_idx = torch.argmax(dim_logits).item()
                predicted_action_indices[i] = action_idx
                predicted_action_one_hot[start_idx + action_idx] = 1.0

            # Step environment
            try:
                 step_output = env.step(predicted_action_indices)
                 # Handle different environment return formats (gym vs gymnasium)
                 if len(step_output) == 4: # Older gym style (obs, rew, done, info)
                    next_state, reward, terminated, info = step_output
                    truncated = False # Assume not truncated if not returned
                 elif len(step_output) == 5: # Newer gymnasium style (obs, rew, terminated, truncated, info)
                    next_state, reward, terminated, truncated, info = step_output
                 else:
                    logger.error(f"Unexpected environment step return format: {step_output}")
                    break # End episode

                 next_state = np.array(next_state, dtype=np.float32)
                 reward = float(reward)
                 done = terminated or truncated

                 if next_state.shape != (STATE_DIM,):
                    logger.error(f"Next state shape mismatch at step {ep_len}! Expected ({STATE_DIM},), got {next_state.shape}. Check env.step().")
                    break # End episode

            except Exception as e:
                 logger.error(f"Environment step failed at step {ep_len}: {e}", exc_info=True)
                 break # End episode on error

            # Update context buffers
            try:
                norm_next_state = (next_state - state_mean) / state_std
                context_states.append(norm_next_state)
                context_actions_one_hot.append(predicted_action_one_hot)
                current_target_rtg -= reward
                context_rtgs.append(np.array([current_target_rtg], dtype=np.float32))
                current_timestep = context_timesteps[-1].item() + 1 # Get scalar and increment
                context_timesteps.append(np.array([min(current_timestep, max_ep_len - 1)], dtype=np.int64)) # Store as array

                # Remove oldest entries (FIFO queue)
                context_states.pop(0)
                context_actions_one_hot.pop(0)
                context_rtgs.pop(0)
                context_timesteps.pop(0)
            except Exception as e:
                 logger.error(f"Error updating context buffers at step {ep_len}: {e}", exc_info=True)
                 break # End this episode

            ep_return += reward
            ep_len += 1
            state = next_state

            if ep_len >= max_ep_len * 1.5: # More generous safety break
                 logger.warning(f"Episode {ep+1} exceeded max length {max_ep_len} significantly ({ep_len} steps), terminating.")
                 done = True

        # --- End of episode loop ---
        if ep_len > 0: # Record only if the episode ran at least one step
            episode_returns.append(ep_return)
            episode_lengths.append(ep_len)
            logger.info(f"Episode {ep+1}/{num_episodes} Finished: Return={ep_return:.2f}, Length={ep_len}")
        else:
            logger.warning(f"Episode {ep+1}/{num_episodes} failed to start or run any steps.")


    # --- End of all episodes loop ---
    env.close() # Close the environment

    if not episode_returns:
        logger.warning("No episodes completed successfully during evaluation.")
        return None, None

    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    mean_length = np.mean(episode_lengths)
    logger.info("-" * 30)
    logger.info(f"Evaluation Results (Target RTG: {target_rtg}):")
    logger.info(f"Episodes Evaluated: {len(episode_returns)}")
    logger.info(f"Mean Return: {mean_return:.2f} +/- {std_return:.2f}")
    logger.info(f"Mean Length: {mean_length:.2f}")
    logger.info("-" * 30)
    return mean_return, std_return

# --- Environment Creator Functions ---

def create_real_env(graphics=False):
    """Creates the actual SolidEnvironmentGameObs environment."""
    if not ENV_AVAILABLE:
        raise ImportError("SolidEnvironmentGameObs not available. Cannot create real environment.")
    logger.info(f"Creating real environment (Graphics: {graphics})...")
    # Adjust parameters as needed for evaluation
    env = SolidEnvironmentGameObs(
        seed_value=random.randint(0, 10000), # Use random seed for eval variability
        graphics=graphics,
        weight=0, # Weight might not be relevant for DT eval, depends on env logic
        logging=False,
        path=os.path.join("Builds", "MS_Solid", "racing.exe") # Ensure path is correct
    )
    return env

def create_dummy_env():
        """Creates a dummy environment for testing purposes."""
        logger.warning("Using DUMMY environment for evaluation.")
        class DummyEnv:
            def __init__(self):
                self.state_shape = (STATE_DIM,)
                self.action_space_dims = NUM_ACTION_DIMS
                self.observation_space = type('space', (), {'shape': self.state_shape})()
                self.action_space = type('space', (), {'shape': (self.action_space_dims,)})()
                self._step_count = 0
                self._max_steps = 200

            def reset(self, seed=None):
                self._step_count = 0
                state = np.random.rand(*self.state_shape).astype(np.float32)
                return state, {}

            def step(self, action):
                if not isinstance(action, np.ndarray) or action.shape != (self.action_space_dims,):
                     print(f"DummyEnv Warning: Invalid action shape: {action.shape if hasattr(action, 'shape') else type(action)}")
                self._step_count += 1
                next_state = np.random.rand(*self.state_shape).astype(np.float32)
                reward = random.uniform(0.05, 0.15)
                terminated = self._step_count >= self._max_steps
                truncated = False
                info = {}
                return next_state, reward, terminated, truncated, info

            def close(self):
                pass # Dummy env requires no closing action
        return DummyEnv()

# --- Main Execution Block ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate a Decision Transformer model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name suffix of the model to evaluate (e.g., 'PPO_Blended_arousal_SolidObs_DT').")
    parser.add_argument("--target_rtg", type=float, required=True, help="Target Return-to-Go for evaluation.")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate.")
    parser.add_argument("--graphics", action='store_true', help="Run evaluation with environment graphics enabled.")
    parser.add_argument("--dummy_env", action='store_true', help="Use a dummy environment instead of the real one.")

    args = parser.parse_args()

    # Construct the full path to the model directory
    final_model_path = os.path.join('examples', 'Agents', 'DT', 'Results', f"{args.model_name}-final")

    # Choose environment creator
    if args.dummy_env:
        env_creator = create_dummy_env
    elif ENV_AVAILABLE:
        # Use lambda to pass graphics argument
        env_creator = lambda: create_real_env(graphics=args.graphics)
    else:
        logger.error("Real environment not available and --dummy_env not specified. Cannot proceed.")
        exit(1) # Exit if no suitable environment can be created

    # Run evaluation
    evaluate_online(
        env_creator=env_creator,
        model_path=final_model_path,
        target_rtg=args.target_rtg,
        num_episodes=args.num_episodes
    )