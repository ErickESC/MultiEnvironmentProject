import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from NoiseyLinear import NoisyLinear
from mlagents_envs.environment import UnityEnvironment as UE
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
from collections import deque, namedtuple
import random
import numpy as np
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CnnQNetwork(nn.Module):
    def __init__(self,observations,actions):
        super(CnnQNetwork,self).__init__()
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, stride=4)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=4)
        self.l1 = NoisyLinear(2, 128)
        self.l3 = NoisyLinear(384, 256)
        self.fc_adv = NoisyLinear(256, 128)
        self.fc_value = NoisyLinear(256, 128)
        self.adv = NoisyLinear(128, actions)
        self.value = NoisyLinear(128,1)
    def forward(self,y,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        y = F.relu(self.l1(y))
        x = x.view(y.size(0), -1)
        x = torch.cat((x,y),dim=1)
        x = F.relu(self.l3(x))
        x_adv = F.relu(self.fc_adv(x))
        x_adv = self.adv(x_adv)
        x_value = F.relu(self.fc_value(x))
        x_value = self.value(x_value)
        advAverage = torch.mean(x_adv,dim=1,keepdim=True)
        output = x_value + x_adv - advAverage

        return output
    def reset_noise(self):
        self.l1.reset_noise()
        self.l3.reset_noise()
        self.fc_adv.reset_noise()
        self.fc_value.reset_noise()
        self.adv.reset_noise()
        self.value.reset_noise()

class QNetwork(nn.Module):
    def __init__(self,observations,actions):
        super(QNetwork,self).__init__()
        self.input = NoisyLinear(observations, 512)
        self.hl = NoisyLinear(512, 256)
        self.hl2 = NoisyLinear(256, 128)
        self.fc_adv = NoisyLinear(128, 128)
        self.fc_value = NoisyLinear(128, 128)
        self.adv = NoisyLinear(128, actions)
        self.value = NoisyLinear(128,1)
    def forward(self, x):
        if x.ndim < 2:
            x = x.view(1,-1)
        x = self.input(x)
        x = F.relu(x)
        x = F.relu(self.hl(x))
        x = F.relu(self.hl2(x))
        x_adv = F.relu(self.fc_adv(x))
        x_adv = self.adv(x_adv)
        x_value = F.relu(self.fc_value(x))
        x_value = self.value(x_value)
        advAverage = x_adv.mean(dim=1, keepdim=True)
        output = x_value + x_adv - advAverage
        return output
    def reset_noise(self):
        self.input.reset_noise()
        self.fc_adv.reset_noise()
        self.fc_value.reset_noise()
        self.adv.reset_noise()
        self.value.reset_noise()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class replayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
import torch
import numpy as np

class ReplayMemory(object):
    def __init__(self, size, batch_size, state_shape, priority=False, alpha=0.99, beta_start=0.4, beta_frames=100000000, epsilon=1e-6):
        self.size = size
        self.batch_size = batch_size
        self.index = 0
        self.full = False
        self.priority_replay = priority
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames
        self.epsilon = epsilon

        self.state = torch.zeros((size, state_shape), dtype=torch.float32)
        self.action = torch.zeros(size, dtype=torch.int64)
        self.reward = torch.zeros(size, dtype=torch.float32)
        self.next_state = torch.zeros((size, state_shape), dtype=torch.float32)
        self.done = torch.zeros(size, dtype=torch.int64)

        if priority:
            self.priorities = torch.ones(size, dtype=torch.float32)

    def push(self, state, action, next_state, reward, done):
        self.state[self.index] = state
        self.action[self.index] = action
        self.reward[self.index] = reward
        self.next_state[self.index] = next_state
        self.done[self.index] = done

        if self.priority_replay:
            self.priorities[self.index] = self.priorities.max().item()

        self.index += 1
        if self.index >= self.size:
            self.full = True
            self.index = 0

    def sample(self):
        max_index = self.size if self.full else self.index

        if self.priority_replay:
            priorities = self.priorities[:max_index].numpy()
            probs = priorities ** self.alpha
            probs /= probs.sum()

            indexes = np.random.choice(max_index, self.batch_size, p=probs)
            total = max_index
            self.beta = min(1.0, self.beta + self.beta_increment)
            weights = (total * probs[indexes]) ** (-self.beta)
            weights /= weights.max()
            weights = torch.tensor(weights, dtype=torch.float32)
        else:
            indexes = np.random.choice(max_index, self.batch_size, replace=False)
            weights = torch.ones(self.batch_size, dtype=torch.float32)

        states = self.state[indexes]
        actions = self.action[indexes]
        next_states = self.next_state[indexes]
        rewards = self.reward[indexes]
        dones = self.done[indexes]

        return (states, actions, next_states, rewards, dones, weights, indexes)

    def update_priorities(self, indexes, td_errors):
        if not self.priority_replay:
            return
        td_errors = torch.abs(td_errors) + self.epsilon
        self.priorities[indexes] = td_errors.cpu().detach()

    def __len__(self):
        return self.size if self.full else self.index

        




class DQN():
    def __init__(self,action_space, state_space, continue_learning=False,input_model="",
                epsilon=0.3, epsilon_decay = True,double_DQN = True, CNN = True):
        self.cnn = CNN
        self.lr = 1e-5
        self.double_DQN = double_DQN
        self.df = 0.99
        self.action_space = action_space
        self.state_space = state_space
        if self.cnn:
            self.policy_net = CnnQNetwork(actions=action_space, observations=state_space).to(device)
            self.target_net = CnnQNetwork(actions=action_space, observations=state_space).to(device)
        else:
            self.policy_net = QNetwork(actions=action_space, observations=state_space).to(device)
            self.target_net = QNetwork(actions=action_space, observations=state_space).to(device)
        if continue_learning:
            self.policy_net.load_state_dict(torch.load(input_model, weights_only=True))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.episode_epsilon_start = 0
        self.episode_epsilon = 0
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon
        self.epsilon = self.epsilon_start
        self.epsilon_decay_rate = 1000000
        self.min_epsilon = 0.02
        self.replay_memory_size = 1000000
        self.netword_sync_rate = 10000
        self.steps = 0
        self.max_steps = 1
        self.tau = 0.05
        self.mini_batch_size = 64
        self.memory = ReplayMemory(self.replay_memory_size,self.mini_batch_size,self.state_space,priority=True)
        self.memory_index = 0
        self.sync_count = 0
        self.update_rate = 4
        self.update_count = 0
    def save_model(self,num):
        print("hi")
        torch.save(self.policy_net.state_dict(), f"CarSnake{num}.pt")
    def act(self,state):
        rand = np.random.rand()
        if rand <= self.epsilon:
            action = np.random.randint(0,self.action_space)
        else:
            with torch.no_grad():
                action = self.policy_net(state)
                action = action.argmax().item()
        return action
    def noisy_act(self,state):
        self.policy_net.reset_noise()
        with torch.no_grad():
            action = self.policy_net(state)
            action = action.argmax().item()
        return action
    def cnn_act(self,state):
        rand = np.random.rand()
        if rand <= self.epsilon:
            action = np.random.randint(0,self.action_space)
        else:
            image, vector = state
            with torch.no_grad():
                action = self.policy_net(image.unsqueeze(0), vector.unsqueeze(0))
                action = action.argmax().item()
        return action
    def noisy_cnn_act(self,state):
        image, vector = state
        with torch.no_grad():
            self.policy_net.reset_noise()
            action = self.policy_net(image.unsqueeze(0), vector.unsqueeze(0))
            action = action.argmax().item()
        return action
    
    def optimize(self,mini_batch):
        transitions = mini_batch
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.mini_batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * self.df) + reward_batch

        loss = torch.mean((expected_state_action_values.unsqueeze(1) - state_action_values)**2).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def cnn_optimize_double(self, mini_batch):
        transitions = mini_batch
        batch = Transition(*zip(*transitions))
        state_imgs = torch.stack([s[0] for s in batch.state]).to(device)
        state_vecs = torch.stack([s[1] for s in batch.state]).to(device)

        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)

        non_final_next_imgs = torch.stack([s[0] for s in batch.next_state if s is not None]).to(device)
        non_final_next_vecs = torch.stack([s[1] for s in batch.next_state if s is not None]).to(device)
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        state_action_values = self.policy_net(state_imgs, state_vecs).gather(1, action_batch)

        next_state_values = torch.zeros(self.mini_batch_size, device=device)

        with torch.no_grad():
            next_policy_q = self.policy_net(non_final_next_imgs, non_final_next_vecs)
            best_actions = next_policy_q.argmax(1, keepdim=True)

            next_target_q = self.target_net(non_final_next_imgs, non_final_next_vecs)
            selected_target_q = next_target_q.gather(1, best_actions).squeeze(1)

            next_state_values[non_final_mask] = selected_target_q

        expected_state_action_values = reward_batch + (self.df * next_state_values)

        loss = torch.mean((expected_state_action_values.unsqueeze(1) - state_action_values)**2)

        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()


    def optimize_double(self,mini_batch):
        self.policy_net.reset_noise()
        self.target_net.reset_noise()
        state_batch = mini_batch[0].to(device)
        action_batch = mini_batch[1].to(device).unsqueeze(1)
        next_state_batch = mini_batch[2].to(device)
        reward_batch = mini_batch[3].to(device).unsqueeze(1)
        dones = mini_batch[4].to(device).unsqueeze(1)
        weights = mini_batch[5].to(device).unsqueeze(1)
        indexes = mini_batch[6]
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            best_actions = self.policy_net(next_state_batch).argmax(1, keepdim=True)
            selected_target_q = self.target_net(next_state_batch).gather(1, best_actions)
            expected_state_action_values = reward_batch + (self.df * selected_target_q * (1-dones))
        # TD-error
        td_errors = expected_state_action_values - state_action_values

        # Weighted MSE loss (with PER)
        loss = (td_errors.pow(2) * weights).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 20.0)
        self.optimizer.step()
        td_errors = td_errors.detach().squeeze()
        if td_errors.ndim > 1:
            td_errors = td_errors.squeeze(1)
        # Update priorities in replay buffer
        self.memory.update_priorities(indexes, td_errors)

    def train(self, episodes, environment: UE):
        behaviour_name = list(environment.behavior_specs)[0]
        self.policy_net.train()
        self.target_net.train()
        j = 1

        for i in range(episodes):
            total_rewards = 0
            self.steps_episode = 0
            self.episode_epsilon = 0
            environment.reset()
            decision_steps, terminal_steps = environment.get_steps(behaviour_name)
            
            state = self.from_numpy(decision_steps[0].obs).to(device)
            tracked_agent = decision_steps[0].agent_id

            while tracked_agent not in terminal_steps.agent_id:
                self.epsilon = self.min_epsilon + (self.epsilon_start - self.min_epsilon) * \
                    math.exp(-1. * self.steps / self.epsilon_decay_rate)
                self.steps += 1
                
                #self.policy_net.reset_noise()
                action = self.noisy_act(state)
                unity_action = ActionTuple(discrete=np.array([[action]], dtype=np.int32))
                environment.set_action_for_agent(behaviour_name, tracked_agent, unity_action)

                environment.step()
                decision_steps, terminal_steps = environment.get_steps(behaviour_name)

                action_tensor = torch.tensor([[action]], dtype=torch.int64, device=device)

                if tracked_agent in terminal_steps:
                    reward = terminal_steps[0].reward
                    new_state = self.from_numpy(decision_steps[0].obs).to(device)
                    self.memory.push(state, int(action), new_state, float(reward),1)
                elif tracked_agent in decision_steps:
                    reward = decision_steps[0].reward
                    new_state = self.from_numpy(decision_steps[0].obs).to(device)
                    
                    self.memory.push(state, int(action), new_state, float(reward),0)
                    state = new_state

                if self.steps % self.update_rate == 0:
                    if len(self.memory) > 1000:
                        mini_batch = self.memory.sample()
                        if self.double_DQN:
                            self.optimize_double(mini_batch)
                        else:
                            self.optimize(mini_batch)

                    # Soft update
                    #target_net_state_dict = self.target_net.state_dict()
                    #policy_net_state_dict = self.policy_net.state_dict()
                    #for key in policy_net_state_dict:
                    #    target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
                    #self.target_net.load_state_dict(target_net_state_dict)
                if reward > 0:
                    total_rewards += 1

                if self.steps % 100000 == 0:
                    self.update_count += 1
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            if i % 100 == 0:
                print("hi")
                self.save_model(j)
                j += 1
            print(f"Episode: {i}, Total Rewards: {total_rewards}, Total Updates:{self.update_count}")
    def cnn_train(self, episodes, environment: UE):
        behaviour_name = list(environment.behavior_specs)[0]
        self.policy_net.train()
        self.target_net.train()

        for i in range(episodes):
            total_rewards = 0
            self.steps_episode = 0
            self.episode_epsilon = 0
            environment.reset()
            decision_steps, terminal_steps = environment.get_steps(behaviour_name)
            goal_ray = torch.from_numpy(decision_steps[0].obs[0]).float().to(device)
            image = (torch.from_numpy(decision_steps[0].obs[1]).float()/255).to(device)
            state = [image,goal_ray]
            tracked_agent = decision_steps[0].agent_id
            self.policy_net.reset_noise()

            while tracked_agent not in terminal_steps.agent_id:
                self.epsilon = self.min_epsilon + (self.epsilon_start - self.min_epsilon) * \
                    math.exp(-1. * self.steps / self.epsilon_decay_rate)
                self.steps += 1
                action = self.noisy_cnn_act((image, goal_ray))
                unity_action = ActionTuple(discrete=np.array([[action]], dtype=np.int32))
                environment.set_action_for_agent(behaviour_name, tracked_agent, unity_action)

                environment.step()
                decision_steps, terminal_steps = environment.get_steps(behaviour_name)

                action_tensor = torch.tensor([[action]], dtype=torch.int64, device=device)

                if tracked_agent in terminal_steps:
                    reward = terminal_steps[0].reward
                    reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)
                    self.memory.push(state, action_tensor, None, reward_tensor)
                elif tracked_agent in decision_steps:
                    reward = decision_steps[0].reward
                    new_goal_ray = torch.from_numpy(decision_steps[0].obs[0]).float().to(device)
                    new_image = (torch.from_numpy(decision_steps[0].obs[1]).float()/255).to(device)
                    new_state = [new_image,new_goal_ray]
                    
                    reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)
                    self.memory.push(state, action_tensor, new_state, reward_tensor)
                    #print("Reward:", reward_tensor)
                    state = new_state

                if self.steps % self.update_rate == 0:
                    if len(self.memory) > self.mini_batch_size:
                        mini_batch = self.memory.sample(self.mini_batch_size)
                        if self.double_DQN:
                            self.cnn_optimize_double(mini_batch)
                        else:
                            self.optimize(mini_batch)

                    ## Soft update
                    #target_net_state_dict = self.target_net.state_dict()
                    #policy_net_state_dict = self.policy_net.state_dict()
                    #for key in policy_net_state_dict:
                    #    target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
                    #self.target_net.load_state_dict(target_net_state_dict)
                
                total_rewards += reward

                if self.steps % 10000 == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            if self.steps_episode > self.max_steps:
                self.max_steps = self.steps_episode

            print(f"Episode: {i}, Total Rewards: {total_rewards}")
            #time.sleep(0.1)
    def test(self, episodes, environment: UE):
        behaviour_name = list(environment.behavior_specs)[0]
        self.policy_net.eval()
        self.target_net.eval()

        for _ in range(episodes):
            environment.reset()
            decision_steps, terminal_steps = environment.get_steps(behaviour_name)
            
            state = self.from_numpy(decision_steps[0].obs).to(device)
            tracked_agent = decision_steps[0].agent_id

            while tracked_agent not in terminal_steps.agent_id:
                #self.policy_net.reset_noise()
                action = self.noisy_act(state)
                unity_action = ActionTuple(discrete=np.array([[action]], dtype=np.int32))
                environment.set_action_for_agent(behaviour_name, tracked_agent, unity_action)

                environment.step()
                decision_steps, terminal_steps = environment.get_steps(behaviour_name)


                if tracked_agent in decision_steps:
                    new_state = self.from_numpy(decision_steps[0].obs).to(device)
                    state = new_state

    def from_numpy(self,state) -> torch.Tensor:
        if len(state) > 1:
            np_state = state[0]
            np_state = np.append(np_state,state[1])

            input_tensor = torch.from_numpy(np_state).float()
        else:
            state = np.array(state)
            input_tensor = torch.from_numpy(state).float()
        return input_tensor
    
    def cnn_from_numpy(self,state) -> torch.Tensor:
        state = np.reshape(state, (1,70))

        input_tensor = torch.from_numpy(state).float()
        return input_tensor
    def noisy_check(self):
        state = torch.randn(1, self.state_space).to(device)
        self.policy_net.reset_noise()
        q1 = self.policy_net(state)
        self.policy_net.reset_noise()
        q2 = self.policy_net(state)
        print(q1, q2)

try:
    # Create the side channel
    engine_configuration_channel = EngineConfigurationChannel()

    # Set parameters BEFORE launching the environment
    engine_configuration_channel.set_configuration_parameters(
        time_scale=200,             # Run faster than real time
        target_frame_rate=-1         # Unlimited frame rate
    )
    env = UE(file_name="Games/CarSnake200Speed/Racing Game.exe", no_graphics=True, side_channels=[engine_configuration_channel],timeout_wait=60)
    env.reset()

    #print(env.behavior_specs)
    #print("\n")
    behaviour_name = list(env.behavior_specs)[0]
    #print(behaviour_name)
    #print("\n")
    spec = env.behavior_specs[behaviour_name]
    #print(spec)
    #print("\n")
    print(spec.action_spec)
    print(spec.action_spec[1][0])
    print(spec.observation_specs)
    agent = DQN(action_space=spec.action_spec[1][0], state_space=808, continue_learning=False, input_model="CarSnake3.pt", double_DQN=True, CNN=False)
    agent.train(episodes=500000, environment=env)
    #agent.policy_net.train()  # noise only works in train() mode
    #for i in range(5):
    #    agent.policy_net.reset_noise()  # regenerate noise
    #    dummy_input = torch.randn(1, agent.state_space).to(device)
    #    output = agent.policy_net(dummy_input)
    #    print(output)
except KeyboardInterrupt:
    try:
        agent.save_model(0)
        env.close()
    finally:
        pass
finally:
    try:
        agent.save_model(0)
    finally:
        env.close()