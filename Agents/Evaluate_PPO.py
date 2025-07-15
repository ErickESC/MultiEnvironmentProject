import numpy as np
from stable_baselines3 import PPO

from affectively_environments.envs.base import compute_confidence_interval
from affectively_environments.envs.solid_game_obs import SolidEnvironmentGameObs
from affectively_environments.envs.solid_cv import SolidEnvironmentCV

from sb3_contrib import RecurrentPPO

import numpy as np
import pickle

if __name__ == "__main__":

    preference_task = True
    classification_task = False
    weight = 0
    # reward_type = 'arousal'
    reward_type = 'score'
    # reward_type = 'blended'
    
    if weight == 0:
        label = 'Optimize'
    elif weight == 0.5:
        label = 'Blended'
    else:
        label = 'Arousal'

    model_path = f'examples\\Agents\\PPO\\results\\ppo_{label}.zip'

    # env = SolidEnvironmentCV(0, graphics=True, weight=weight, logging=False, path="Builds\\MS_Solid\\racing.exe")
    env = SolidEnvironmentGameObs(0, graphics=True, weight=weight, logging=False, path="Builds\\MS_Solid\\racing.exe")
    model = PPO("MlpPolicy", tensorboard_log="Tensorboard", device='cpu', env=env)
    model.load(model_path)
    model.set_parameters(model_path)
    
    arousal, scores = [], []
    observations, actions, rewards, dones = [], [], [], []

    episode = 0
    for _ in range(5): # Number of episodes to run
        state = env.reset()
        episode_observations, episode_actions, episode_rewards, episode_dones = [], [], [], []
        for i in range(100): # Number of steps to run in each episode
            action, _ = model.predict(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            # Append the data to the episode lists
            episode_observations.append(state)
            episode_actions.append(action)
            if reward_type == 'score':
                episode_rewards.append(env.current_score)
            else:
                episode_rewards.append(reward)
            episode_dones.append(done)
            
            state = next_state
            if done:
                break
        
        episode += 1    
        print(f"Episode: {episode}")
        
        # Append episode data to the main lists
        observations.append(episode_observations)
        actions.append(episode_actions)
        rewards.append(episode_rewards)
        dones.append(episode_dones)

        arousal.append(np.mean(env.arousal_trace))
        scores.append(env.best_score)
        env.best_score = 0
        env.arousal_trace.clear()
    env.close()

    # Save the dataset to a file
    dataset = {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'dones': dones
    }
    print(f"Scores size: {len(scores)}")
    print(f"Scores: {scores}")
    # with open(f'examples\\Agents\\PPO\\datasets\\testing_dataset.pkl', 'wb') as f:
    # # with open(f'examples\\Agents\\PPO\\datasets\\PPO_{label}_{reward_type}_SolidObs_dataset.pkl', 'wb') as f:
    #     pickle.dump(dataset, f)
        
    # print(f"Dataset loaded: {dataset['observations'][0]}")
    # print(f"Dataset loaded: {dataset['rewards'][0]}")
    # print(f"Dataset loaded: {dataset['rewards'][1]}")
    # print(f"Dataset loaded: {dataset['rewards'][2]}")
    # for reward in dataset['rewards']:
    #     print(f"Episode rewards: {reward}")
    # print(f"Dataset loaded: {len(dataset['observations'])}")
    # print(f"Dataset loaded: {len(dataset['observations'][0])}")
    # for obs in dataset["observations"]:
    #     print(f"Trajectory length: {len(obs)}")
    #     print(f"Episode observations: {obs}")
    # --- Suggested TARGET_RETURN ---

    average_reward = np.mean(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)

    print(f"\nSuggested TARGET_RETURN (average reward): {average_reward:.2f}")
    print(f"Consider experimenting with values between {min_reward:.2f} and {max_reward:.2f}")
    print(f"Best Score: {compute_confidence_interval(scores)}, Mean Arousal: {compute_confidence_interval(arousal)}")
    print(f"Done creating {label}-{reward_type} dataset")
