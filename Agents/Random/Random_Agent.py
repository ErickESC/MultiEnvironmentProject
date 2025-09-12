# from affectively_environments.envs.pirates import Pirates_Environment
from affectively_environments.envs.solid_game_obs import SolidEnvironmentGameObs
from affectively_environments.envs.base import compute_confidence_interval
import numpy as np
import pickle

if __name__ == "__main__":

    weight = 0
    env = SolidEnvironmentGameObs(0, graphics=False, weight=weight, logging=False, path="Builds\\MS_Solid\\racing.exe")
    
    arousal, scores = [], []
    observations, actions, rewards, dones = [], [], [], []
    
    episode = 0
    for _ in range(30): # Number of episodes to run
        _ = env.reset()
        episode_observations, episode_actions, episode_rewards, episode_dones = [], [], [], []
        for i in range(600): # Number of steps to run in each episode
            action = env.action_space.sample()
            _, reward, done, info = env.step(action)
            # Append the data to the episode lists
            episode_observations.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_dones.append(done)
            if done:
                state = env.reset()
        arousal.append(np.mean(env.arousal_trace))
        scores.append(env.best_score)
        env.best_score = 0
        env.arousal_trace.clear()
        episode += 1    
        print(f"Episode: {episode}")
        
        # Append episode data to the main lists
        observations.append(episode_observations)
        actions.append(episode_actions)
        rewards.append(episode_rewards)
        dones.append(episode_dones)
    env.close()
    
    # Save the dataset to a file
    dataset = {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'dones': dones
    }
    with open(f'examples\\Agents\\PPO\\datasets\\testing_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Best Score: {compute_confidence_interval(scores)}, Mean Arousal: {compute_confidence_interval(arousal)}")
