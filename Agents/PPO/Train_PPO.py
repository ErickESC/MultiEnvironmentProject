import numpy as np
import sys

from stable_baselines3 import PPO

from affectively_environments.envs.pirates import PiratesEnvironment
from affectively_environments.envs.solid_game_obs import SolidEnvironmentGameObs
from affectively_environments.envs.solid_cv import SolidEnvironmentCV

from stable_baselines3.common.callbacks import CheckpointCallback


def main(run, weight, env_type):
    np.set_printoptions(suppress=True, precision=6)

    if env_type.lower() == "pirates":
        env = PiratesEnvironment(id_number=run, weight=weight, graphics=False, logging=True,
                                 path="Builds\\MS_Pirates\\platform.exe", log_prefix="PPO/")
    elif env_type.lower() == "solid":
        env = SolidEnvironmentGameObs(id_number=run, weight=weight, graphics=False, logging=True,
                                      path="Builds\\MS_Solid\\racing.exe", log_prefix="PPO/")
    else:
        raise ValueError("Invalid environment type. Choose 'pirates' or 'solid'.")

    # env = SolidEnvironmentGameObs(id_number=run, weight=weight, graphics=False, logging=True,
    #                               path="Builds\\MS_Solid\\racing.exe", log_prefix="PPO/")
    
    sideChannel = env.customSideChannel
    env.targetSignal = np.ones

    if weight == 0:
        label = 'optimize'
    elif weight == 0.5:
        label = 'blended'
    else:
        label = 'arousal'

    # Create the callback
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=f"examples\\Agents\\PPO\\checkpoints\\PPO",
        name_prefix=f"ppo_{env_type.lower()}_obs_{label}_{run}"
    )

    model = PPO("MlpPolicy", env=env, tensorboard_log="Tensorboard", device='cpu')
    model.learn(total_timesteps=1000000, progress_bar=True, callback=checkpoint_callback)
    model.save(f"examples\\Agents\\PPO\\results\\ppo_{env_type.lower()}_obs_{label}_{run}")


if __name__ == "__main__":
    # if len(sys.argv) != 4:
    #     print("Usage: python script.py <run-number> <weight> <environment>")
    #     sys.exit(1)

    # try:
    #     run = int(sys.argv[1])
    #     weight = float(sys.argv[2])
    #     env_type = sys.argv[3]
    # except ValueError as e:
    #     print(f"Error in argument parsing: {e}")
    #     sys.exit(1)
    run = 3
    weight = 1
    env_type = "solid"  # or "pirates"

    main(run, weight, env_type)

