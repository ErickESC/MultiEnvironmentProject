import numpy as np
import onnxruntime as ort
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import pandas as pd
def add_to_csv(data, filename):

    df = pd.DataFrame(data)
    df.to_csv(filename, mode='a', header=False, index=False)
# Path to Unity environment and ONNX model
UNITY_ENV_PATH = "Games/3DBallHard/UnityEnvironment.exe"
ONNX_MODEL_PATH = "MultiEnvironmentProject/ML_Agent/Models/ONNX/3DBallHard.onnx"
try:
    engine_configuration_channel = EngineConfigurationChannel()

    # Set parameters BEFORE launching the environment
    engine_configuration_channel.set_configuration_parameters(
        time_scale=1000,             # Run faster than real time
        target_frame_rate=60         # Unlimited frame rate
    )
    # Start Unity environment
    env = UnityEnvironment(file_name=UNITY_ENV_PATH, seed=1, side_channels=[engine_configuration_channel],no_graphics=True)
    env.reset()

    # Load ONNX model
    session = ort.InferenceSession(ONNX_MODEL_PATH)
    input_name1 = session.get_inputs()[0].name  # usually 'obs_0'
    
    if len(session.get_inputs())>1:
        input_name2 = session.get_inputs()[1].name  # usually 'obs_0'
        print(input_name2)
    print(session.get_inputs())
    print(input_name1)

    # Get behavior name
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]

    print(f"[INFO] Behavior: {behavior_name}")
    #print(f"[INFO] Observation shapes: {[o.shape for o in spec.observation_shapes]}")
    print(f"[INFO] Action space: {'continuous' if spec.action_spec.is_continuous() else 'discrete'}")
    step_count = 0
    episode_actions = []
    episode_rewards = []
    episode_observations = []
    # Main loop
    for episode in range(5):  # Run a few episodes
        episodic_step_count = 0
        env.reset()
        actions = []
        observations = []
        rewards = []
        agent_ids = []
        while True:
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            # Collect actions

            for agent_id in decision_steps.agent_id:
                obs_list = decision_steps[agent_id].obs
                #print("Observation:", obs_list)
                #obs = np.concatenate([o.flatten() for o in obs_list], axis=0).astype(np.float32)
                #obs = obs.reshape(1, -1)  # shape (1, 8) for 3DBall
                observations.append(obs_list)
                reward = decision_steps[agent_id].reward
                rewards.append(reward)

                # Run ONNX inference
                if len(session.get_inputs())>1:
                    action = session.run(None, {input_name1: obs_list[0].reshape(1,-1), input_name2:obs_list[1].reshape(1,-1)})  # shape (1, 2)
                else:
                    action = session.run(None, {input_name1: np.array([obs_list[0]],dtype=np.float32)})  # shape (1, 2)
                #print("Action:",action)
                action = action[2]
                actions.append(action[0])  # shape (2,)
                agent_ids.append(agent_id)
                data = {
                    "observations": [[obs_list[i].tolist() for i in range(len(obs_list))]],
                    "actions": [action.tolist()],
                    "reward" : reward,
                    "game" : behavior_name.split('?')[0],
                    "episode": episode+1,
                    "time_step": episodic_step_count,
                    "agent_id": agent_id
                }
                add_to_csv(data, "MultiEnvironmentProject/Database/MLAgentsOutputs.csv")


            # Convert and set actions
            # Convert and set actions
            #action_array = np.array(actions, dtype=np.float32)  # shape (N_agents, 2)
            #print(action_array)  # Check shape
                action_tuple = ActionTuple(continuous=action)
                env.set_action_for_agent(behavior_name, agent_id, action_tuple)


            env.step()
            step_count += 1
            episodic_step_count += 1

            # Exit episode when agents are done
            if len(terminal_steps) > 0:
                break
        
        episode_actions.append(actions)
        episode_rewards.append(rewards)
        episode_observations.append(observations)


    env.close()
except KeyboardInterrupt:
    env.close()
    print("[INFO] Process interrupted by user.")
finally:
    print("Episode rewards:", episode_rewards)
    if 'env' in locals():
        env.close()
    print("[INFO] Environment closed.")