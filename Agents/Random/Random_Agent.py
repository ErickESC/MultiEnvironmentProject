# from affectively_environments.envs.pirates import Pirates_Environment
from affectively_environments.envs.solid_game_obs import SolidEnvironmentGameObs
from affectively_environments.envs.base import compute_confidence_interval
import numpy as np

if __name__ == "__main__":

    weight = 1
    env = SolidEnvironmentGameObs(0, graphics=False, weight=weight, logging=False, path="Builds\\MS_Solid\\racing.exe")
    
    arousal, scores = [], []
    episode = 0
    for _ in range(30):
        _ = env.reset()
        for i in range(600):
            action = env.action_space.sample()
            _, reward, done, info = env.step(action)
            if done:
                state = env.reset()
        arousal.append(np.mean(env.arousal_trace))
        scores.append(env.best_score)
        env.best_score = 0
        env.arousal_trace.clear()
        episode += 1    
        print(f"Episode: {episode}")
    env.close()
    
    print(f"Best Score: {compute_confidence_interval(scores)}, Mean Arousal: {compute_confidence_interval(arousal)}")
