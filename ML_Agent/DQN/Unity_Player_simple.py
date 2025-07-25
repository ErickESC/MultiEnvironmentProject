from mlagents_envs.environment import UnityEnvironment as UE

env = UE(file_name="Games/WallJumpInference/UnityEnvironment.exe", no_graphics=False, side_channels=[],timeout_wait=60)
env.reset()

behavior_name = list(env.behavior_specs)[0]

for _ in range(1000):
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    
    for agent_id in decision_steps.agent_id:
        obs_list = decision_steps[agent_id].obs
        print(f"Agent {agent_id} observation 0:", obs_list[0])  # Print first sensor
        # You can also log obs, reward, etc.

    # NOTE: You are not calling set_actions or affecting the agent in any way