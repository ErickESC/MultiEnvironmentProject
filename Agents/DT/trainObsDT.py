import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    DecisionTransformerConfig,
    DecisionTransformerModel,
    Trainer,
    TrainingArguments,
)
import numpy as np
import pickle
import logging
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import random

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Data Loading and Preprocessing ---

def load_data(pickle_paths,
                   cumulative_rewards_in_dataset=False):
    """
    Loads dataset from a pickle file generated.

    Args:
        pickle_path (str): Path to the dataset .pkl file.
        cumulative_rewards_in_dataset (bool): Set to True if the 'rewards' list
                                              in the pickle file contains the
                                              cumulative score up to that timestep.
                                              Set to False if it contains the
                                              per-step reward.

    Returns:
        List[List[Dict[str, Any]]]: A list of episodes.
        Each episode is a list of transitions (dictionaries).
        Example transition:
        {
            'observation': np.array([...]),
            'action': np.array([a1, a2, a3]),
            'reward': float, # <-- This will be the PER-STEP reward
            'done': bool
        }
    """
    processed_episodes = []
    all_final_scores = [] # To calculate average return from data
    for pickle_path in pickle_paths:
        logger.info(f"Loading dataset from: {pickle_path}")
        try:
            with open(pickle_path, 'rb') as f:
                dataset = pickle.load(f)
        except FileNotFoundError:
            logger.error(f"Dataset file not found at: {pickle_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading pickle file: {e}")
            raise

        # Extract data lists (List of Lists)
        observations_list = dataset['observations']
        actions_list = dataset['actions']
        rewards_list = dataset['rewards'] # This might be cumulative or per-step
        dones_list = dataset['dones']

        num_episodes = len(observations_list)
        if not (len(actions_list) == num_episodes and len(rewards_list) == num_episodes and len(dones_list) == num_episodes):
            logger.error("Dataset lists (observations, actions, rewards, dones) have inconsistent lengths.")
            raise ValueError("Inconsistent number of episodes in dataset file.")

        # Add logic to choose the best n trajectories based on total reward
        observations_list, actions_list, rewards_list, dones_list = choose_best_episodes(observations_list, actions_list, rewards_list, 
                                                                                         dones_list, cumulative_rewards_in_dataset=cumulative_rewards_in_dataset)



        num_episodes = len(observations_list) # Update to the filtered number of episodes

        logger.info(f"Processing {num_episodes} episodes...")
        for i in range(num_episodes):
            ep_obs = observations_list[i]
            ep_act = actions_list[i]
            ep_rew = rewards_list[i] # Raw rewards from file
            ep_done = dones_list[i]

            ep_len = len(ep_obs)
            if not (len(ep_act) == ep_len and len(ep_rew) == ep_len and len(ep_done) == ep_len):
                logger.warning(f"Episode {i} has inconsistent lengths between obs/act/rew/done. Skipping. Lengths: O={len(ep_obs)}, A={len(ep_act)}, R={len(ep_rew)}, D={len(ep_done)}")
                continue
            if ep_len == 0:
                logger.warning(f"Episode {i} is empty. Skipping.")
                continue

            episode_data = []
            current_episode_total_reward = 0.0
            for t in range(ep_len):
                 # --- Input Validation and Correction ---
                # Ensure observation is numpy array of correct shape and type
                try:
                    observation = np.array(ep_obs[t], dtype=np.float32)
                    if observation.shape != (STATE_DIM,):
                        if observation.shape[0] < STATE_DIM:
                            observation = pad_dimensions(observation,STATE_DIM)
                        else:
                            raise ValueError(f"Observation shape mismatch: expected ({STATE_DIM},), got {observation.shape}")
                except Exception as e:
                    logger.warning(f"Skipping step {t} in episode {i} due to invalid observation: {e}")
                    continue # Skip this step if observation is bad

                # Ensure action is numpy array of correct shape, type, and value range
                try:
                    action = np.array(ep_act[t], dtype=np.int64)
                    if action.shape != (NUM_ACTION_DIMS,):
                        if action.shape[0] < NUM_ACTION_DIMS:
                            action = pad_dimensions(action, NUM_ACTION_DIMS)
                        else:
                            raise ValueError(f"Action shape mismatch: expected ({NUM_ACTION_DIMS},), got {action.shape}")
                    # Check if action values are within bounds
                    for dim_idx, act_val in enumerate(action):
                        if not (-100 <= act_val < NUM_ACTIONS_PER_DIM[dim_idx]):
                            raise ValueError(f"Action value out of bounds in dim {dim_idx}: got {act_val}, expected < {NUM_ACTIONS_PER_DIM[dim_idx]}")
                except Exception as e:
                     logger.warning(f"Skipping step {t} in episode {i} due to invalid action: {e}")
                     continue # Skip this step if action is bad

                done = bool(ep_done[t])

                # --- Calculate PER-STEP reward ---
                step_reward = 0.0
                raw_reward_t = ep_rew[t] # The value stored at this step in the file

                if cumulative_rewards_in_dataset:
                    if t == 0:
                        step_reward = float(raw_reward_t) # First step's reward is the first cumulative value
                    else:
                        # Per-step reward is diff between current and previous cumulative score
                        raw_reward_prev = ep_rew[t-1]
                        # Handle potential type issues during subtraction
                        try:
                            step_reward = float(raw_reward_t) - float(raw_reward_prev)
                        except (TypeError, ValueError) as e:
                             logger.warning(f"Could not calculate step reward at step {t} in episode {i} due to type issue: {e}. Setting reward to 0.")
                             step_reward = 0.0
                else:
                    # If dataset stores per-step rewards directly
                    try:
                        step_reward = float(raw_reward_t)
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Could not parse step reward at step {t} in episode {i} due to type issue: {e}. Setting reward to 0.")
                        step_reward = 0.0


                current_episode_total_reward += step_reward

                # Append the transition with the calculated *per-step* reward
                episode_data.append({
                    'observation': observation,
                    'action': action, # Store the original integer action indices
                    'reward': step_reward,
                    'done': done
                })
            # --- End of episode loop ---

            # Only add episode if it contains valid data
            if episode_data:
                processed_episodes.append(episode_data)
                all_final_scores.append(current_episode_total_reward) # Store the calculated total reward
            else:
                 logger.warning(f"Episode {i} resulted in no valid steps after cleaning. Skipping.")

        # --- End of all episodes loop ---

        if not processed_episodes:
            logger.error("No valid episodes found after processing.")
            raise ValueError("No episodes processed. Check dataset integrity and input validation steps.")

        avg_return = np.mean(all_final_scores) if all_final_scores else 0
        max_return = np.max(all_final_scores) if all_final_scores else 0
        logger.info(f"Loaded and processed {len(processed_episodes)} episodes.")
        logger.info(f"Calculated Average Episode Return from data: {avg_return:.2f}")
        logger.info(f"Calculated Max Episode Return from data: {max_return:.2f}")
        logger.info(f"Suggest using TARGET_RETURN around {avg_return:.2f} for evaluation.")
    random.shuffle(processed_episodes)
    return processed_episodes


def calculate_returns_to_go(episode):
    """Calculates returns-to-go for a single episode."""
    rewards = np.array([step['reward'] for step in episode])
    n = len(rewards)
    rtgs = np.zeros_like(rewards, dtype=np.float32)
    cumulative_reward = 0
    for t in reversed(range(n)):
        cumulative_reward += rewards[t]
        rtgs[t] = cumulative_reward
    return rtgs

def process_dataset(episodes, context_length):
    """Processes raw episodes into sequences suitable for Decision Transformer."""
    all_sequences = []
    max_ep_len = 0
    all_states = []

    logger.info("Processing episodes and calculating RTGs...")
    for episode_idx, episode in enumerate(episodes):
        if not episode: # Skip empty episodes that might slip through
            logger.warning(f"Skipping empty episode at index {episode_idx} in process_dataset.")
            continue
        ep_len = len(episode)
        max_ep_len = max(max_ep_len, ep_len)
        rtgs = calculate_returns_to_go(episode)

        # Collect all states for normalization
        for step in episode:
            all_states.append(step['observation'])

        # Create sequences
        for t in range(ep_len):
            # Sequence ends at index t, target is t+1
            start_idx = max(0, t - context_length + 1)
            end_idx = t + 1 # Slices are exclusive at the end

            # Extract subsequences
            seq_states = [step['observation'] for step in episode[start_idx:end_idx]]
            seq_actions = [step['action'] for step in episode[start_idx:end_idx]] # Action indices
            seq_rtgs = rtgs[start_idx:end_idx]
            seq_timesteps = np.arange(start_idx, end_idx)

            # --- Target ---
            # Target action INDICES are the ones *after* the sequence ends (at index t+1)
            if t < ep_len - 1:
                target_action = episode[t+1]['action'] # The action indices taken *after* state s_t
            else:
                 continue # Skip the last state as we can't predict a next action

            # --- Padding ---
            padding_len = context_length - len(seq_states)

            padded_states = np.concatenate([np.zeros((padding_len, STATE_DIM), dtype=np.float32), np.array(seq_states, dtype=np.float32)], axis=0)
            # Pad actions with -100 (original indices)
            padded_actions = np.concatenate([np.full((padding_len, NUM_ACTION_DIMS), -100, dtype=np.int64), np.array(seq_actions, dtype=np.int64)], axis=0)
            padded_rtgs = np.concatenate([np.zeros(padding_len, dtype=np.float32), seq_rtgs], axis=0)
            padded_timesteps = np.concatenate([np.zeros(padding_len, dtype=np.int64), seq_timesteps], axis=0)
            attention_mask = np.concatenate([np.zeros(padding_len, dtype=np.int64), np.ones(len(seq_states), dtype=np.int64)], axis=0)

            all_sequences.append({
                "states": padded_states,
                "actions": padded_actions, # Store padded action INDICES (int64, uses -100 padding)
                "returns_to_go": padded_rtgs.reshape(-1, 1), # Needs shape (K, 1)
                "timesteps": padded_timesteps.reshape(-1, 1), # Needs shape (K, 1)
                "attention_mask": attention_mask,
                "targets": target_action # Target action INDICES (NUM_ACTION_DIMS,)
            })

    logger.info(f"Created {len(all_sequences)} sequences.")
    logger.info(f"Maximum episode length found: {max_ep_len}")

    # Calculate state normalization statistics from training data only
    if not all_states:
         raise ValueError("No states collected for normalization. Check data processing.")
    all_states_np = np.array(all_states)
    state_mean = np.mean(all_states_np, axis=0)
    state_std = np.std(all_states_np, axis=0) + 1e-6 # Add epsilon for stability

    # Normalize states in the sequences
    for seq in all_sequences:
        # Normalize only non-padded states
        valid_state_mask = seq['attention_mask'] == 1
        seq['states'][valid_state_mask] = (seq['states'][valid_state_mask] - state_mean) / state_std
        # Ensure padded states remain zero AFTER normalization of valid states
        seq['states'][~valid_state_mask] = 0.0


    return all_sequences, max_ep_len, state_mean, state_std

def choose_best_episodes(observations_list, actions_list, rewards_list, dones_list, cumulative_rewards_in_dataset=True):
    filtered_observations, filtered_actions, filtered_rewards, filtered_dones = [], [], [], []
    # Calculate the total rewards for each episode
    # Calculate total rewards for each episode
    total_rewards = []
    rewards_list = np.array(rewards_list)
    B,T = rewards_list.shape[0], rewards_list.shape[1]
    rewards_list = rewards_list.reshape(B,T)
    for rewards in rewards_list:
        if cumulative_rewards_in_dataset:
            # If rewards are cumulative, the last reward in the list is the total reward
            total_rewards.append(rewards[-1] if rewards else 0)
        else:
            # If rewards are per-step, sum them to get the total reward
            total_rewards.append(sum(rewards))

    # Sort episodes by total reward in descending order
    sorted_indices = np.argsort(total_rewards)[::-1]

    # Select the top PICK_BEST episodes
    selected_indices = sorted_indices[:PICK_BEST]

    # Filter the lists based on selected indices
    filtered_observations = [observations_list[i] for i in selected_indices]
    filtered_actions = [actions_list[i] for i in selected_indices]
    filtered_rewards = [rewards_list[i] for i in selected_indices]
    filtered_dones = [dones_list[i] for i in selected_indices]
    
    return filtered_observations, filtered_actions, filtered_rewards, filtered_dones

# --- 2. Hugging Face Dataset and Collator ---

class DecisionTransformerDataset(TorchDataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = self.sequences[idx]
        # Convert numpy arrays to torch tensors
        # Ensure actions and targets are LongTensors for CrossEntropyLoss
        return {
            "states": torch.tensor(item["states"], dtype=torch.float32),
            "actions": torch.tensor(item["actions"], dtype=torch.int64), # Action indices should be long
            "returns_to_go": torch.tensor(item["returns_to_go"], dtype=torch.float32),
            "timesteps": torch.tensor(item["timesteps"], dtype=torch.int64),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.int64),
            "targets": torch.tensor(item["targets"], dtype=torch.int64) # Target action indices should be long
        }

# Custom Data Collator
@dataclass
class DecisionTransformerDataCollator:
    def __call__(self, features):
        batch = {}
        # Stack tensors from individual sequences
        batch["states"] = torch.stack([f["states"] for f in features])
        batch["actions"] = torch.stack([f["actions"] for f in features]) # Action indices (long)
        batch["returns_to_go"] = torch.stack([f["returns_to_go"] for f in features])
        batch["timesteps"] = torch.stack([f["timesteps"] for f in features])
        batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
        batch["targets"] = torch.stack([f["targets"] for f in features]) # Target action indices (long)
        return batch

# --- 3. Model Configuration and Custom Trainer for Loss ---
# Custom Trainer to handle multi-discrete action loss
class MultiDiscreteDecisionTransformerTrainer(Trainer):
    def __init__(self, *, num_actions_per_dim=None, **kwargs):
        super().__init__(**kwargs)
        if num_actions_per_dim is None:
            raise ValueError("`num_actions_per_dim` must be provided for MultiDiscreteDecisionTransformerTrainer")
        self.num_actions_per_dim = num_actions_per_dim
        self.num_action_dims = len(num_actions_per_dim)
        self.action_slice_starts = np.concatenate(([0], np.cumsum(self.num_actions_per_dim)[:-1]))
        self.concatenated_action_dim = sum(self.num_actions_per_dim)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        
        _ = kwargs # Avoid unused variable warning

        # Keep original action INDICES (long tensor) for loss calculation
        original_actions_long = inputs["actions"] # Shape: (batch, seq_len, num_action_dims)
        original_actions_long_mask = (original_actions_long != -100)
        original_actions_long[~original_actions_long_mask] = 0

        # --- Create concatenated one-hot actions for MODEL INPUT ---
        batch_size, seq_len, _ = original_actions_long.shape
        one_hot_actions_list = []
        for i in range(self.num_action_dims):
            action_dim_indices = original_actions_long[:, :, i].clone()
            padding_mask_dim = (action_dim_indices == -100)
            action_dim_indices[padding_mask_dim] = 0 # Replace padding with 0 for one_hot
            one_hot_dim = F.one_hot(action_dim_indices, num_classes=self.num_actions_per_dim[i])
            one_hot_dim[padding_mask_dim] = 0 # Ensure padded actions are zero vectors
            one_hot_actions_list.append(one_hot_dim)
        actions_for_model_input = torch.cat(one_hot_actions_list, dim=-1).float()
        # --- End one-hot creation ---

        # --- Prepare Timesteps for MODEL INPUT ---
        timesteps_for_model_input = inputs["timesteps"].squeeze(-1) # Remove the trailing dimension
        # --- End Timesteps Preparation ---

        # Forward pass
        outputs = model(
            states=inputs["states"],
            actions=actions_for_model_input,
            returns_to_go=inputs["returns_to_go"],
            timesteps=timesteps_for_model_input,
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )

        action_preds = outputs.action_preds # Logits output by the model

        # --- Align Predictions and Targets for Loss ---
        act_dim = action_preds.shape[-1]
        all_logits = action_preds.view(-1, act_dim)

        target_actions_for_loss = original_actions_long.long()

        valid_action_mask = (target_actions_for_loss != -100).all(dim=-1)
        valid_action_mask_flat = valid_action_mask.view(-1)

        valid_logits = all_logits[valid_action_mask_flat]

        if valid_logits.shape[0] == 0:
             logger.warning("No valid actions found in batch for loss calculation. Returning 0 loss.")
             loss_val = torch.tensor(0.0, device=model.device, requires_grad=True)
             return (loss_val, outputs) if return_outputs else loss_val

        valid_targets_indices = target_actions_for_loss[valid_action_mask]

        total_loss = 0
        criterion = nn.CrossEntropyLoss()

        for i in range(self.num_action_dims):
            start_idx = self.action_slice_starts[i]
            end_idx = start_idx + self.num_actions_per_dim[i]
            dim_logits = valid_logits[:, start_idx:end_idx]
            dim_targets = valid_targets_indices[:, i].long()
            loss = criterion(dim_logits, dim_targets)
            total_loss += loss

        avg_loss = total_loss / self.num_action_dims

        return (avg_loss, outputs) if return_outputs else avg_loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform prediction step for evaluation. Overridden to handle custom input preparation.
        """
        if ignore_keys is None:
            ignore_keys = []

        # Manually prepare inputs for the model, similar to compute_loss
        # Ensure 'targets' is not passed directly to model(**model_inputs)
        original_actions_long = inputs["actions"] # Keep for loss/label comparison if needed

        # Create one-hot actions for model input
        batch_size, seq_len, _ = original_actions_long.shape
        one_hot_actions_list = []
        for i in range(self.num_action_dims):
            action_dim_indices = original_actions_long[:, :, i].clone()
            padding_mask_dim = (action_dim_indices == -100)
            action_dim_indices[padding_mask_dim] = 0
            one_hot_dim = F.one_hot(action_dim_indices, num_classes=self.num_actions_per_dim[i])
            one_hot_dim[padding_mask_dim] = 0
            one_hot_actions_list.append(one_hot_dim)
        actions_for_model_input = torch.cat(one_hot_actions_list, dim=-1).float()

        # Prepare timesteps
        timesteps_for_model_input = inputs["timesteps"].squeeze(-1)

        # Prepare the dictionary of inputs the model forward pass expects
        model_inputs = {
            "states": inputs["states"],
            "actions": actions_for_model_input,
            "returns_to_go": inputs["returns_to_go"],
            "timesteps": timesteps_for_model_input,
            "attention_mask": inputs["attention_mask"],
            "return_dict": True, # Ensure we get outputs dictionary
        }

        # Standard prediction_step logic from Trainer, but using prepared inputs
        loss = None
        logits = None
        labels = None # We'll define labels based on our problem context

        with torch.no_grad(): # Evaluation is done without gradients
            # Call model with prepared inputs
            outputs = model(**model_inputs)

            # Calculate loss if not prediction_loss_only
            if not prediction_loss_only:
                # Use compute_loss logic to calculate loss based on model outputs and original actions
                action_preds = outputs.action_preds
                act_dim = action_preds.shape[-1]
                all_logits = action_preds.view(-1, act_dim)

                target_actions_for_loss = original_actions_long.long()

                valid_action_mask = (target_actions_for_loss != -100).all(dim=-1)
                valid_action_mask_flat = valid_action_mask.view(-1)

                valid_logits = all_logits[valid_action_mask_flat]

                if valid_logits.shape[0] > 0:
                    valid_targets_indices = target_actions_for_loss[valid_action_mask]
                    total_loss = 0
                    criterion = nn.CrossEntropyLoss()
                    for i in range(self.num_action_dims):
                        start_idx = self.action_slice_starts[i]
                        end_idx = start_idx + self.num_actions_per_dim[i]
                        dim_logits = valid_logits[:, start_idx:end_idx]
                        dim_targets = valid_targets_indices[:, i].long()
                        loss_dim = criterion(dim_logits, dim_targets)
                        total_loss += loss_dim
                    loss = total_loss / self.num_action_dims
                else:
                    loss = torch.tensor(0.0, device=model.device)

            # Logits are the direct output from the model's prediction head
            logits = outputs.action_preds # Shape: (batch, seq_len, concatenated_action_dim)

            # Labels: What should we compare against? For sequence prediction like DT,
            # the 'labels' are typically the actions we are trying to predict at each step.
            # In our case, the loss compares predictions at step `t` with actions `a_t`.
            labels = original_actions_long # Shape: (batch, seq_len, num_action_dims)

        # Detach tensors before returning, standard practice in prediction_step
        if loss is not None:
            loss = loss.detach()
        if logits is not None:
            logits = logits.detach()
        if labels is not None:
            labels = labels.detach()

        return (loss, logits, labels)


# --- 4. Training and compute metrics function ---

def train(pickle_path, cumulative_rewards_in_dataset=False, model_name='decision-transformer', 
          training_args=None, model_dir=None):

    # 1. Load and Process Data
    raw_episodes = load_data(pickle_path, cumulative_rewards_in_dataset)
    # Split episodes (simple sequential split here, consider shuffling episodes first)
    split_idx = int(len(raw_episodes) * 0.9)
    train_episodes = raw_episodes[:split_idx]
    eval_episodes = raw_episodes[split_idx:]

    # Process Training Data (get normalization stats)
    train_sequences, max_ep_len_train, state_mean, state_std = process_dataset(
        train_episodes, CONTEXT_LENGTH
    )
    if not train_sequences:
         raise ValueError("Training data processing resulted in zero sequences.")

    # Process Evaluation Data (use training stats)
    eval_sequences, max_ep_len_eval, _, _ = process_dataset(
        eval_episodes, CONTEXT_LENGTH # Pass eval episodes
    )
    # Re-normalize eval states with training stats
    if eval_sequences: # Check if eval set is not empty
        for seq in eval_sequences:
            valid_state_mask = seq['attention_mask'] == 1
            # Perform normalization carefully: subtract mean, divide by std
            seq['states'][valid_state_mask] = (seq['states'][valid_state_mask] - state_mean) / state_std
            # Ensure padded states remain zero AFTER normalization of valid states
            seq['states'][~valid_state_mask] = 0.0
    else:
        logger.warning("Evaluation dataset processing resulted in zero sequences.")


    max_ep_len = max(max_ep_len_train, max_ep_len_eval) if max_ep_len_eval is not None else max_ep_len_train
    if max_ep_len == 0:
        max_ep_len = 500 # Provide a sensible default if no episodes were processed
        logger.warning(f"Max episode length was 0, setting to default: {max_ep_len}")


    train_dataset = DecisionTransformerDataset(train_sequences)
    eval_dataset = DecisionTransformerDataset(eval_sequences) if eval_sequences else None # Handle empty eval

    # 2. Configure Model
    # act_dim is the size of the concatenated one-hot action vector
    concatenated_action_dim = sum(NUM_ACTIONS_PER_DIM) # 3 + 3 + 2 = 8

    config = DecisionTransformerConfig(
        state_dim=STATE_DIM,
        act_dim=concatenated_action_dim, # Size of the input action embedding (e.g., 8)
        hidden_size=128,
        n_layer=3,
        n_head=4,
        activation_function="relu",
        dropout=0.1,
        n_inner=None,
        max_length=CONTEXT_LENGTH,
        max_ep_len=max_ep_len, # Crucial for position embeddings
        action_tanh=False, # Should be False for discrete/one-hot actions
    )
    model = DecisionTransformerModel(config)
    model.to(DEVICE)

    collator = DecisionTransformerDataCollator()

    trainer = MultiDiscreteDecisionTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, # Pass eval_dataset (can be None)
        data_collator=collator,
        num_actions_per_dim=NUM_ACTIONS_PER_DIM, # Pass action space details
        compute_metrics=compute_metrics,
    )

    # 4. Train
    logger.info("Starting training...")
    trainer.train()

    # 5. Save final model
    final_model_path = os.path.join(model_dir, "Results", f"{model_name}_final")
    trainer.save_model(final_model_path)

    # Save normalization stats
    np.savez(os.path.join(final_model_path, "normalization_stats.npz"),
             mean=state_mean, std=state_std, max_ep_len=max_ep_len)
    
    logger.info("Training finished.")
    return model, state_mean, state_std, max_ep_len, trainer

def compute_metrics(eval_pred):
    """
    Computes evaluation metrics (accuracy, F1 score) for multi-discrete predictions.
    Expects eval_pred to be a tuple: (loss, logits, labels)
    """
    logits, labels = eval_pred
    # Convert tensors to numpy arrays if necessary.
    if hasattr(logits, "cpu"):
        logits = logits.cpu().numpy()
    if hasattr(labels, "cpu"):
        labels = labels.cpu().numpy()
    batch_size, seq_len, concat_dim = logits.shape
    num_action_dims = labels.shape[-1]
    
    predictions = []
    all_accs = []
    all_preds = []
    all_targets = []
    
    # For each action dimension, compute the predicted actions
    start = 0
    for i in range(num_action_dims):
        num_classes = NUM_ACTIONS_PER_DIM[i]
        end = start + num_classes
        # Get predictions for this dimension: shape (batch, seq_len)
        pred_i = logits[:, :, start:end].argmax(axis=-1)
        target_i = labels[:, :, i]
        # Only consider positions where target is valid (not padding)
        valid_mask = (target_i != -100)
        valid_preds = pred_i[valid_mask]
        valid_targets = target_i[valid_mask]
        # Compute accuracy for this dimension if we have any valid targets
        if valid_targets.size > 0:
            acc = (valid_preds == valid_targets).mean()
            try:
                f1 = f1_score(valid_targets, valid_preds, average="micro")
            except Exception:
                f1 = 0.0
        else:
            acc = 0.0
            f1 = 0.0
        all_accs.append(acc)
        all_preds.append(valid_preds)
        all_targets.append(valid_targets)
        start = end

    # Overall metrics computed by concatenating all predictions (across dims)
    if all_preds and np.concatenate(all_preds).size > 0:
        overall_f1 = f1_score(np.concatenate(all_targets), np.concatenate(all_preds), average="micro")
    else:
        overall_f1 = 0.0
    overall_accuracy = np.mean(all_accs) if all_accs else 0.0

    return {"eval_accuracy": overall_accuracy, "eval_f1": overall_f1}

def plot_metrics(log_history, output_dir):
    """
    Extracts metrics from the trainer log history and plots them per epoch.

    Args:
        log_history (list): List of logged dicts from trainer.state.log_history.
        output_dir (str): Directory where the plot image will be saved.
    """

    epochs = []
    train_loss = []
    eval_acc = []
    eval_f1 = []

    for log in log_history:
        if "epoch" in log:
            epochs.append(log["epoch"])
            train_loss.append(log.get("loss", None))
            eval_acc.append(log.get("eval_accuracy", None))
            eval_f1.append(log.get("eval_f1", None))

    # Filter out entries without evaluation metrics (if any)
    filtered = [(e, tl, acc, f1)
                for e, tl, acc, f1 in zip(epochs, train_loss, eval_acc, eval_f1)]
    if not filtered:
        print("No evaluation metrics available for plotting.")
        return

    epochs, train_loss, eval_acc, eval_f1 = zip(*filtered)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Training Loss", marker="o")
    plt.plot(epochs, eval_acc, label="Eval Accuracy", marker="o")
    plt.plot(epochs, eval_f1, label="Eval F1", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Training Metrics per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "training_metrics.png")
    
    plt.savefig(plot_path)
    plt.show()
    logger.info(f"Training metrics plot saved to {plot_path}")

def pad_dimensions(input_vector, dims):
    return np.pad(input_vector, (0,dims-input_vector.shape[0]), mode="constant", constant_values=-100)

# --- Constants and Hyperparameters ---
NUM_ACTIONS_PER_DIM = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]  # Action space for SolidGame environment
NUM_ACTION_DIMS = 10#len(NUM_ACTIONS_PER_DIM)
STATE_DIM = 400 # State dimension for SolidGame environment 54 continuous 7 discrete
CONTEXT_LENGTH = 20  # K: How many steps the model sees
TARGET_RETURN = 8.71 # Target return for evaluation
PICK_BEST = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Root path for server
ROOT_PATH = os.path.join(os.sep, "home", "eser22eo", "Affectively-Framework")

# --- Main Execution ---
if __name__ == "__main__":

    weight = 0.5
    # reward_type = 'arousal'
    # reward_type = 'score'
    reward_type = 'blended'
    
    # data_from = 'PPO'
    data_from = 'Explore'
    
    # running_on_server = True
    running_on_server = False

    if weight == 0:
        label = 'Optimize'
    elif weight == 0.5:
        label = 'Blended'
    else:
        label = 'Arousal'

    # --- CHOOSE DATASET ---
    # Set cumulative_rewards_in_dataset=True if 'rewards' is cumulative score
    # Set cumulative_rewards_in_dataset=False if 'rewards' is per-step reward
    CUMULATIVE_REWARDS = True if reward_type == 'score' else False
    
    #dataset_path = os.path.join("examples", "Agents", data_from, "datasets", f"{data_from}_{label}_{reward_type}_moreTrained.pkl") 
    dataset_paths = ["MultiEnvironmentProject/Database/solid_test_dataset.pkl","MultiEnvironmentProject/Database/pirates_test_dataset.pkl"]
    #DIR_PATH = os.path.join("examples", "Agents", "DT")
    DIR_PATH = "MultiEnvironmentProject/Agents/DT"
    # dataset_path = os.path.join("examples", "Agents", data_from, "datasets", f"testing_dataset.pkl")
    
    for dataset_path in dataset_paths:
        if running_on_server:
            dataset_path = os.path.join(ROOT_PATH, "examples", "Agents", data_from, "datasets", f"{data_from}_{label}_{reward_type}_SolidObs_dataset.pkl")
            DIR_PATH = os.path.join(ROOT_PATH, "examples", "Agents", "DT")
            # dataset_path = os.path.join(ROOT_PATH, "examples", "Agents", data_from, "datasets", f"testing_dataset.pkl")

        # Verify that paths are correct
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset directory not found at: {dataset_path}")
            raise SystemExit(f"Dataset directory not found at: {dataset_path}")
        if not os.path.exists(DIR_PATH):
            logger.error(f"Directory not found at: {DIR_PATH}")
            raise SystemExit(f"Directory not found at: {DIR_PATH}")

    
    dt_name = f"{data_from}_{label}_{reward_type}_moreTrained_DT"
    
    # 3. Set up Trainer
    training_args = TrainingArguments(
        output_dir=os.path.join(DIR_PATH, "output", f"{dt_name}_output"),
        overwrite_output_dir=True,
        num_train_epochs=30,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=1e-4,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        logging_dir=os.path.join(DIR_PATH, "logs", f"{dt_name}_logs"),
        logging_steps=100,
        save_strategy="steps",
        save_steps=500,
        # Use eval_steps if eval_dataset exists
        eval_strategy="steps", # if eval_dataset else "no",
        eval_steps=500, # if eval_dataset else None,
        # load_best_model_at_end=True if eval_dataset else False,
        # metric_for_best_model="eval_loss" if eval_dataset else None,
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=False,
        remove_unused_columns=False, # Important for custom compute_loss
    )

    logger.info(f"Traning: {dt_name} on {dataset_path}")

    # Train the model
    trained_model, s_mean, s_std, m_ep_len, trainer = train(pickle_path=dataset_paths, cumulative_rewards_in_dataset=CUMULATIVE_REWARDS, 
                                                            training_args=training_args, model_name=dt_name, model_dir=DIR_PATH)
    
    logger.info(f"Model {dt_name} trained")
    # print(f"State Mean: {s_mean}, State Std: {s_std}, Max Episode Length: {m_ep_len}")
    
    # Save trainer.state.log_history to a JSON file
    log_history_path = os.path.join(DIR_PATH, "Results", f"{dt_name}_final", "log_history.json")
    with open(log_history_path, "w") as json_file:
        json.dump(trainer.state.log_history, json_file, indent=4)

    logger.info(f"Trainer log history saved to {log_history_path}")
    
    model_dir = os.path.join(DIR_PATH, "Results", f"{dt_name}_final")
    # plot_metrics(trainer.state.log_history, model_dir)