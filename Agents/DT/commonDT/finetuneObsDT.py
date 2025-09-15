import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import (
    DecisionTransformerConfig,
    DecisionTransformerModel,
    Trainer,
    TrainingArguments,
)
import numpy as np
import pickle
import random
import logging
import os
import time # For timestamping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from affectively_environments.envs.solid_game_obs import SolidEnvironmentGameObs

# --- Configuration (Adjust these as needed) ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper Function (Copy-pasted from previous script, with slight mods) ---
# (load_your_data, calculate_returns_to_go, process_dataset,
#  DecisionTransformerDataset, DecisionTransformerDataCollator,
#  MultiDiscreteDecisionTransformerTrainer - these will be reused or adapted)

# --- Re-using Data Processing Functions from the main training script ---
# Make sure these functions are defined as in your original training script.
# For brevity, I'll assume they are available.
# If they are in a separate file, import them:
# from your_training_script import (
#     load_your_data, calculate_returns_to_go, process_dataset,
#     DecisionTransformerDataset, DecisionTransformerDataCollator,
#     MultiDiscreteDecisionTransformerTrainer
# )

def calculate_returns_to_go(episode_rewards_list):
    """Calculates returns-to-go for a single episode's rewards list."""
    rewards = np.array(episode_rewards_list, dtype=np.float32)
    n = len(rewards)
    rtgs = np.zeros_like(rewards)
    cumulative_reward = 0
    for t in reversed(range(n)):
        cumulative_reward += rewards[t]
        rtgs[t] = cumulative_reward
    return rtgs.tolist()


def process_dataset_for_finetuning(episodes, context_length, state_mean, state_std):
    """
    Processes raw episodes into sequences suitable for Decision Transformer.
    Uses provided state_mean and state_std for normalization.
    This is adapted from the original process_dataset.
    """
    all_sequences = []
    max_ep_len = 0
    # all_states = [] # Not needed if mean/std are provided

    logger.info("Processing new episodes for fine-tuning...")
    for episode_idx, episode_data in enumerate(episodes):
        # episode_data is expected to be a list of dicts:
        # [{'observation': s, 'action': a, 'reward': r, 'done': d, 'rtg': rtg_val_at_step_t, 'timestep': t_val}, ...]
        # The RTG here is the target RTG used during collection, NOT the true future sum.
        # We will recalculate true RTGs for training.

        ep_len = len(episode_data)
        if ep_len == 0:
            logger.warning(f"Skipping empty episode at index {episode_idx}.")
            continue
        max_ep_len = max(max_ep_len, ep_len)

        # Extract per-step rewards to calculate true RTGs for this new trajectory
        ep_rewards = [step['reward'] for step in episode_data]
        true_rtgs_for_episode = calculate_returns_to_go(ep_rewards) # List of RTGs

        # Create sequences
        for t in range(ep_len):
            start_idx = max(0, t - context_length + 1)
            end_idx = t + 1

            seq_states_raw = [step['observation'] for step in episode_data[start_idx:end_idx]]
            # Normalize states
            seq_states_normalized = [(s - state_mean) / (state_std + 1e-6) for s in seq_states_raw]

            seq_actions = [step['action'] for step in episode_data[start_idx:end_idx]] # Action indices
            seq_true_rtgs = true_rtgs_for_episode[start_idx:end_idx] # Use true RTGs for training
            seq_timesteps = np.arange(start_idx, end_idx)

            if t < ep_len - 1:
                target_action = episode_data[t+1]['action']
            else:
                 continue

            padding_len = context_length - len(seq_states_normalized)

            padded_states = np.concatenate([np.zeros((padding_len, STATE_DIM), dtype=np.float32), np.array(seq_states_normalized, dtype=np.float32)], axis=0)
            padded_actions = np.concatenate([np.full((padding_len, NUM_ACTION_DIMS), -100, dtype=np.int64), np.array(seq_actions, dtype=np.int64)], axis=0)
            padded_rtgs = np.concatenate([np.zeros(padding_len, dtype=np.float32), np.array(seq_true_rtgs, dtype=np.float32)], axis=0)
            padded_timesteps = np.concatenate([np.zeros(padding_len, dtype=np.int64), seq_timesteps], axis=0)
            attention_mask = np.concatenate([np.zeros(padding_len, dtype=np.int64), np.ones(len(seq_states_normalized), dtype=np.int64)], axis=0)

            # Ensure padded states are 0 AFTER normalization of valid parts
            padded_states[attention_mask == 0] = 0.0

            all_sequences.append({
                "states": padded_states,
                "actions": padded_actions,
                "returns_to_go": padded_rtgs.reshape(-1, 1),
                "timesteps": padded_timesteps.reshape(-1, 1),
                "attention_mask": attention_mask,
                "targets": target_action
            })

    logger.info(f"Created {len(all_sequences)} new sequences from fine-tuning data.")
    # max_ep_len from new data might be different, the model's max_ep_len is fixed by pre-training
    return all_sequences


class DecisionTransformerDataset(TorchDataset):
    def __init__(self, sequences):
        self.sequences = sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        item = self.sequences[idx]
        return {
            "states": torch.tensor(item["states"], dtype=torch.float32),
            "actions": torch.tensor(item["actions"], dtype=torch.int64),
            "returns_to_go": torch.tensor(item["returns_to_go"], dtype=torch.float32),
            "timesteps": torch.tensor(item["timesteps"], dtype=torch.int64),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.int64),
            "targets": torch.tensor(item["targets"], dtype=torch.int64)
        }

@dataclass
class DecisionTransformerDataCollator:
    def __call__(self, features):
        batch = {}
        batch["states"] = torch.stack([f["states"] for f in features])
        batch["actions"] = torch.stack([f["actions"] for f in features])
        batch["returns_to_go"] = torch.stack([f["returns_to_go"] for f in features])
        batch["timesteps"] = torch.stack([f["timesteps"] for f in features])
        batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
        batch["targets"] = torch.stack([f["targets"] for f in features])
        return batch

class MultiDiscreteDecisionTransformerTrainer(Trainer):
    def __init__(self, *, num_actions_per_dim=None, **kwargs):
        super().__init__(**kwargs)
        if num_actions_per_dim is None:
            raise ValueError("`num_actions_per_dim` must be provided")
        self.num_actions_per_dim = num_actions_per_dim
        self.num_action_dims = len(num_actions_per_dim)
        self.action_slice_starts = np.concatenate(([0], np.cumsum(self.num_actions_per_dim)[:-1]))
        self.concatenated_action_dim = sum(self.num_actions_per_dim)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        _ = kwargs
        original_actions_long = inputs["actions"]
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
        timesteps_for_model_input = inputs["timesteps"].squeeze(-1)
        outputs = model(
            states=inputs["states"], actions=actions_for_model_input,
            returns_to_go=inputs["returns_to_go"], timesteps=timesteps_for_model_input,
            attention_mask=inputs["attention_mask"], return_dict=True,
        )
        action_preds = outputs.action_preds
        act_dim = action_preds.shape[-1]
        all_logits = action_preds.view(-1, act_dim)
        target_actions_for_loss = original_actions_long.long()
        valid_action_mask = (target_actions_for_loss != -100).all(dim=-1)
        valid_action_mask_flat = valid_action_mask.view(-1)
        valid_logits = all_logits[valid_action_mask_flat]
        if valid_logits.shape[0] == 0:
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
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if ignore_keys is None: ignore_keys = []
        original_actions_long = inputs["actions"]
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
        timesteps_for_model_input = inputs["timesteps"].squeeze(-1)
        model_inputs = {
            "states": inputs["states"], "actions": actions_for_model_input,
            "returns_to_go": inputs["returns_to_go"], "timesteps": timesteps_for_model_input,
            "attention_mask": inputs["attention_mask"], "return_dict": True,
        }
        loss, logits, labels = None, None, None
        with torch.no_grad():
            outputs = model(**model_inputs)
            if not prediction_loss_only:
                action_preds = outputs.action_preds
                act_dim = action_preds.shape[-1]
                all_logits = action_preds.view(-1, act_dim)
                target_actions_for_loss = original_actions_long.long()
                valid_action_mask = (target_actions_for_loss != -100).all(dim=-1)
                valid_action_mask_flat = valid_action_mask.view(-1)
                valid_logits = all_logits[valid_action_mask_flat]
                if valid_logits.shape[0] > 0:
                    valid_targets_indices = target_actions_for_loss[valid_action_mask]
                    total_loss = 0; criterion = nn.CrossEntropyLoss()
                    for i in range(self.num_action_dims):
                        start_idx, end_idx = self.action_slice_starts[i], self.action_slice_starts[i] + self.num_actions_per_dim[i]
                        loss_dim = criterion(valid_logits[:, start_idx:end_idx], valid_targets_indices[:, i].long())
                        total_loss += loss_dim
                    loss = total_loss / self.num_action_dims
                else: loss = torch.tensor(0.0, device=model.device)
            logits = outputs.action_preds; labels = original_actions_long
        if loss is not None: loss = loss.detach()
        if logits is not None: logits = logits.detach()
        if labels is not None: labels = labels.detach()
        return (loss, logits, labels)


def collect_online_data(env_creator, model, model_path_for_stats, target_rtg, num_episodes):
    """Collects data by rolling out the current model in the environment."""
    logger.info(f"Collecting {num_episodes} new episodes with target RTG: {target_rtg}...")

    # Load normalization stats from the (pre-trained) model path
    stats = np.load(os.path.join(model_path_for_stats, "normalization_stats.npz"))
    state_mean = stats['mean']
    state_std = stats['std']
    model_max_ep_len = int(stats['max_ep_len'])

    _num_actions_per_dim = NUM_ACTIONS_PER_DIM # Global
    _num_action_dims = len(_num_actions_per_dim)
    _action_slice_starts = np.concatenate(([0], np.cumsum(_num_actions_per_dim)[:-1]))
    _concatenated_action_dim = sum(_num_actions_per_dim)

    env = env_creator()
    model.to(DEVICE).eval() # Ensure model is in eval mode and on correct device

    collected_episodes_data = [] # List of episodes, each episode is a list of step dicts

    for ep_idx in range(num_episodes):
        state = env.reset()
        state = np.array(state, dtype=np.float32)
        done = False
        current_episode_reward_sum = 0
        episode_timesteps = 0
        current_target_rtg = float(target_rtg)

        # Context buffers for model input
        norm_state = (state - state_mean) / (state_std + 1e-6)
        context_states = [np.zeros_like(state, dtype=np.float32)] * (CONTEXT_LENGTH - 1) + [norm_state]
        context_actions_one_hot = [np.zeros(_concatenated_action_dim, dtype=np.float32)] * CONTEXT_LENGTH
        # Store the *commanded* RTG, not the true future RTG (which we don't know yet)
        context_rtgs = [np.array([current_target_rtg], dtype=np.float32)] * CONTEXT_LENGTH
        context_timesteps_scalar = [0] * CONTEXT_LENGTH # Store scalar timesteps for np.array creation

        current_episode_transitions = []

        while not done:
            # Prepare inputs for the Decision Transformer
            states_tensor = torch.tensor(np.array(context_states)[-CONTEXT_LENGTH:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            actions_tensor = torch.tensor(np.array(context_actions_one_hot)[-CONTEXT_LENGTH:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            rtgs_tensor = torch.tensor(np.array(context_rtgs)[-CONTEXT_LENGTH:], dtype=torch.float32).unsqueeze(0).to(DEVICE)

            timesteps_np = np.array(context_timesteps_scalar[-CONTEXT_LENGTH:], dtype=np.int64)
            timesteps_input_tensor = torch.tensor(timesteps_np, dtype=torch.long).unsqueeze(0).to(DEVICE) # Shape (1, K)

            attn_mask_tensor = torch.ones_like(timesteps_input_tensor).to(DEVICE)

            with torch.no_grad():
                outputs = model(
                    states=states_tensor, actions=actions_tensor, returns_to_go=rtgs_tensor,
                    timesteps=timesteps_input_tensor, attention_mask=attn_mask_tensor, return_dict=True
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
            next_state = np.array(next_state, dtype=np.float32)
            reward = float(reward)
            done = terminated

            # Store transition data (state, action_indices, reward, done, commanded_rtg, timestep)
            current_episode_transitions.append({
                'observation': state, # Store raw state BEFORE normalization
                'action': predicted_action_indices.copy(), # Store the action indices taken
                'reward': reward,
                'done': done,
                'rtg_command': current_target_rtg, # The RTG given to model for this step
                'timestep': episode_timesteps
            })

            # Update context for next step
            state = next_state
            norm_next_state = (state - state_mean) / (state_std + 1e-6)
            context_states.append(norm_next_state)
            context_actions_one_hot.append(predicted_action_one_hot)
            current_target_rtg -= reward # Decrement RTG
            context_rtgs.append(np.array([current_target_rtg], dtype=np.float32))
            episode_timesteps += 1
            # Clamp timestep to model's max_ep_len for embedding layer
            context_timesteps_scalar.append(min(episode_timesteps, model_max_ep_len - 1))


            # Trim context buffers
            if len(context_states) > CONTEXT_LENGTH: context_states.pop(0)
            if len(context_actions_one_hot) > CONTEXT_LENGTH: context_actions_one_hot.pop(0)
            if len(context_rtgs) > CONTEXT_LENGTH: context_rtgs.pop(0)
            if len(context_timesteps_scalar) > CONTEXT_LENGTH: context_timesteps_scalar.pop(0)

            current_episode_reward_sum += reward

            if episode_timesteps >= model_max_ep_len : # Safety break based on model's capacity
                logger.warning(f"Episode {ep_idx+1} reached model_max_ep_len {model_max_ep_len}, terminating.")
                done = True # Ensure 'done' is True if last step was due to this
                # If 'done' was already True from env, it's fine.
                # If this is the *reason* for termination, make sure the last transition reflects this
                if not current_episode_transitions[-1]['done']:
                     current_episode_transitions[-1]['done'] = True

        if current_episode_transitions: # Only add if episode has data
            collected_episodes_data.append(current_episode_transitions)
        logger.info(f"Collected episode {ep_idx+1}/{num_episodes}. Return: {current_episode_reward_sum:.2f}, Length: {episode_timesteps}")

    env.close()
    logger.info(f"Finished collecting data. Total episodes: {len(collected_episodes_data)}")
    return collected_episodes_data


def main_finetune():
    # --- Create Environment ---
    def create_env():
        env = SolidEnvironmentGameObs(0, graphics=False, weight=WEIGHT, logging=False, path="Builds\\MS_Solid\\racing.exe", discretize=False)
        sideChannel = env.customSideChannel
        env.targetSignal = np.ones
        return env

    # 1. Load Pre-trained Model and Config
    logger.info(f"Loading pre-trained model from: {PRETRAINED_MODEL_PATH}")
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        logger.error(f"Pre-trained model path not found: {PRETRAINED_MODEL_PATH}")
        return

    model_config = DecisionTransformerConfig.from_pretrained(PRETRAINED_MODEL_PATH)
    # Ensure act_dim in loaded config matches current one-hot encoding sum
    concatenated_action_dim = sum(NUM_ACTIONS_PER_DIM)
    if model_config.act_dim != concatenated_action_dim:
        logger.warning(f"Loaded model_config.act_dim ({model_config.act_dim}) != "
                       f"current concatenated_action_dim ({concatenated_action_dim}). Overriding.")
        model_config.act_dim = concatenated_action_dim
    # Update max_length if current CONTEXT_LENGTH is different (should match generally)
    if model_config.max_length != CONTEXT_LENGTH:
        logger.warning(f"Loaded model_config.max_length ({model_config.max_length}) != "
                       f"current CONTEXT_LENGTH ({CONTEXT_LENGTH}). Using current.")
        model_config.max_length = CONTEXT_LENGTH


    model = DecisionTransformerModel.from_pretrained(PRETRAINED_MODEL_PATH, config=model_config)
    model.to(DEVICE)

    # Load normalization stats (crucial for new data processing and collection)
    stats = np.load(os.path.join(PRETRAINED_MODEL_PATH, "normalization_stats.npz"))
    state_mean = stats['mean']
    state_std = stats['std']
    # This is the max_ep_len the model was originally trained with, for positional embeddings
    original_model_max_ep_len = int(stats['max_ep_len'])


    # 2. Collect New Online Data
    new_episodes_data = collect_online_data(
        env_creator=create_env,
        model=model,
        model_path_for_stats=PRETRAINED_MODEL_PATH, # Use stats from pre-trained model
        target_rtg=TARGET_RETURN_FOR_COLLECTION,
        num_episodes=NUM_ONLINE_EPISODES_TO_COLLECT
    )

    if not new_episodes_data:
        logger.info("No new data collected. Skipping fine-tuning.")
        return

    # # Optional: Save newly collected data
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # new_data_filename = os.path.join(NEW_DATA_SAVE_PATH_BASE, f"finetune_data_{timestamp}.pkl")
    # os.makedirs(os.path.dirname(new_data_filename), exist_ok=True)
    # with open(new_data_filename, 'wb') as f:
    #     pickle.dump(new_episodes_data, f)
    # logger.info(f"Saved newly collected online data to: {new_data_filename}")


    # 3. Process New Data for Training
    # Use state_mean, state_std from the *original* pre-training for consistency
    new_sequences = process_dataset_for_finetuning(
        new_episodes_data, CONTEXT_LENGTH, state_mean, state_std
    )

    if not new_sequences:
        logger.info("Processing new data resulted in zero sequences. Skipping fine-tuning.")
        return

    # Here, you might also want to load your *original* offline dataset and augment it.
    # For simplicity, this example fine-tunes *only* on the new data.
    # To augment:
    # original_sequences = load_and_process_original_dataset(...)
    # combined_sequences = original_sequences + new_sequences
    # train_dataset = DecisionTransformerDataset(combined_sequences)

    train_dataset = DecisionTransformerDataset(new_sequences)
    # For fine-tuning, often an eval set from the new data is not strictly necessary,
    # or you might evaluate on a held-out portion of original data.
    # For simplicity, we'll skip eval_dataset here, or use a small part of new data.
    eval_dataset = None # Or create a small eval split from new_sequences

    # 4. Fine-tune the Model
    finetuned_model_output_dir = os.path.join(OUTPUT_DIR_BASE, f"{FINETUNED_MODEL_NAME_BASE}_{NEW_VERSION}_{timestamp}")
    finetuned_logs_dir = os.path.join(LOGS_DIR_BASE, f"{FINETUNED_MODEL_NAME_BASE}_{NEW_VERSION}_{timestamp}")

    training_args = TrainingArguments(
        output_dir=finetuned_model_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=FINETUNE_EPOCHS,
        per_device_train_batch_size=FINETUNE_BATCH_SIZE,
        learning_rate=FINETUNE_LEARNING_RATE,
        weight_decay=1e-4, # Keep or adjust
        warmup_ratio=0.1,  # Keep or adjust
        logging_dir=finetuned_logs_dir,
        logging_steps=max(1, len(train_dataset) // (FINETUNE_BATCH_SIZE * 5)), # Log ~5 times per epoch
        save_strategy="steps", # Save at the end of each fine-tuning epoch
        # evaluation_strategy="no" if eval_dataset is None else "epoch",
        # load_best_model_at_end=False if eval_dataset is None else True,
        # metric_for_best_model="eval_loss" if eval_dataset else None, # Needs eval_loss if used
        # evaluation_strategy="steps", # Simpler for pure fine-tuning example
        load_best_model_at_end=False,
        remove_unused_columns=False,
    )

    collator = DecisionTransformerDataCollator()

    # The model's config max_ep_len should be based on original training,
    # as positional embeddings are tied to it.
    # If new data has longer episodes, they'll be truncated or timesteps clamped.
    # The `model_max_ep_len` in `collect_online_data` already handles clamping.
    # The `max_ep_len` in `DecisionTransformerConfig` is for the max sequence length of positional embeddings.
    # It should remain as it was during pre-training unless you are re-initializing those embeddings.

    trainer = MultiDiscreteDecisionTransformerTrainer(
        model=model, # Pass the loaded and potentially config-adjusted model
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        num_actions_per_dim=NUM_ACTIONS_PER_DIM,
    )

    logger.info("Starting fine-tuning...")
    trainer.train()

    # 5. Save Fine-tuned Model and its Normalization Stats
    # final_finetuned_model_path = os.path.join("examples", "Agents", "DT", "Results", "fineTuned", f"{FINETUNED_MODEL_NAME_BASE}-{timestamp}")
    final_finetuned_model_path = os.path.join("examples", "Agents", "DT", "Results", "fineTuned", f"{FINETUNED_MODEL_NAME_BASE}_{NEW_VERSION}")
    # os.makedirs(final_finetuned_model_path, exist_ok=True)
    trainer.save_model(final_finetuned_model_path)
    # Save the *original* normalization stats, as fine-tuning doesn't change them
    np.savez(os.path.join(final_finetuned_model_path, "normalization_stats.npz"),
             mean=state_mean, std=state_std, max_ep_len=original_model_max_ep_len)

    logger.info(f"Fine-tuning finished. Model saved to: {final_finetuned_model_path}")
    
# --- ACTION SPACE CONFIGURATION (Must match pre-trained model and env) ---
NUM_ACTIONS_PER_DIM = [3, 3, 2]
NUM_ACTION_DIMS = len(NUM_ACTIONS_PER_DIM)
STATE_DIM = 54 # Must match environment
CONTEXT_LENGTH = 20  # K: Must match pre-trained model's context length

WEIGHT = 0 # Weight for the reward function (0 for optimize, 1 for arousal, 0.5 for blended)

# --- Paths ---
REWARD_TYPE = "Optimize"  # Can be "Optimize", "Arousal", or "Blended"
OLD_VERSION = "v100"
NEW_VERSION = "v200"  # Version of fine-tuned model
# PRETRAINED_MODEL_PATH = "examples\\Agents\\DT\\Results\\preTrained\\PPO_Optimize_score_SolidObs_DT_final"
PRETRAINED_MODEL_PATH = f"examples\\Agents\\DT\\Results\\fineTuned\\ODT_{REWARD_TYPE}_{OLD_VERSION}"
FINETUNED_MODEL_NAME_BASE = f"ODT_{REWARD_TYPE}"
OUTPUT_DIR_BASE = "examples\\Agents\\DT\\output"
LOGS_DIR_BASE = "examples\\Agents\\DT\\logs"
NEW_DATA_SAVE_PATH_BASE = "examples\\Agents\\DT\\Results\\fineTuned"

# --- Fine-tuning Hyperparameters ---
NUM_ONLINE_EPISODES_TO_COLLECT = 100 # Number of new episodes to collect per fine-tuning iteration
FINETUNE_EPOCHS = 30                # Number of epochs to fine-tune on the augmented dataset
FINETUNE_BATCH_SIZE = 32
FINETUNE_LEARNING_RATE = 5e-5      # Often smaller for fine-tuning
TARGET_RETURN_FOR_COLLECTION = 17  # Target return to use when collecting new data (can be tuned)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    main_finetune()