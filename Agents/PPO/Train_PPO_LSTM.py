from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
import numpy as np
from affectively_environments.envs.solid_cv import SolidEnvironmentCV
from affectively_environments.envs.solid_game_obs import SolidEnvironmentGameObs
from stable_baselines3.common.callbacks import ProgressBarCallback
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
import torch
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, ProgressBarCallback

if __name__ == "__main__":
    run = 20
    weight = 0.5

    # env = SolidEnvironmentCV(
    #     id_number=run,
    #     weight=weight,
    #     graphics=False,
    #     logging=True,
    #     path="Builds\\MS_Solid\\racing.exe",
    #     log_prefix="LSTM/",
    #     grayscale=False
    # )
    
    env = SolidEnvironmentGameObs(
        id_number=run,
        weight=weight,
        graphics=False,
        logging=True,
        path="Builds\\MS_Solid\\racing.exe",
        log_prefix="LSTM/"
    )

    label = 'optimize' if weight == 0 else 'arousal' if weight == 1 else 'blended'

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=500000,
        save_path=f"examples\\Agents\\PPO\\checkpoints\\LSTM",
        name_prefix=f"lstm_ppo_solid_{label}_{run}"
    )
    progress_callback = ProgressBarCallback()
    callbacks = CallbackList([checkpoint_callback, progress_callback])

    model = RecurrentPPO(
        policy="CnnLstmPolicy",
        env=env,
        tensorboard_log="Tensorboard\\LSTM",
        device='cuda',
    )
    model.learn(total_timesteps=5_000_000, callback=callbacks)
    model.save(f"examples\\Agents\\PPO\\lstm_ppo_solid_obs_{label}_{run}_extended")