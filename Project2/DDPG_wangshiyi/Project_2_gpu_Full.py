'''
this code is to examine whether residue netowrks perform better than simple MLPs
'''

# import libs
import gym
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

import imageio # to create gifs

import tqdm

# hyper params
EPISODES = 1000
BUFFER_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.001 # 0.001
LR_ACTOR = 0.0001 # 0.0001
LR_CRITIC = 0.001 # 0.001

env = gym.make('LunarLander-v2', render_mode='rgb_array')

# residue block
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        
        self.lin1 = nn.Linear(in_features, 256)
        self.norm1 = nn.LayerNorm(256)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(256,256)
        self.norm2 = nn.LayerNorm(256)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(256,out_features)
        self.relu3 = nn.ReLU()
        self.norm3 = nn.LayerNorm(out_features)
        
        self.lin_res = nn.Linear(in_features, out_features)
        
        self.relu_final = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.lin1(x)
        out = self.relu1(out)
        out = self.norm1(out)
        out = self.lin2(out)
        out = self.relu2(out)
        out = self.norm2(out)
        out = self.lin3(out)
        out = self.relu3(out)
        out = self.norm3(out)
        
        identity = self.lin_res(identity)
        
        out += identity
        return self.relu_final(out)

# Actor and Critic Network
class Actor(nn.Module):
    def __init__(self,mode=None):
        super(Actor, self).__init__()
        self.mode = mode
        if mode==None:
            raise Exception('Must specify a network mode!')
        if mode=='Linear':
            self.net = nn.Sequential(
                nn.Linear(env.observation_space.shape[0], 64),
                nn.ReLU(),
                nn.Linear(64, env.action_space.n)
            )
        elif mode=='LinearNorm':
            self.net = nn.Sequential(
                nn.Linear(env.observation_space.shape[0], 64),
                nn.ReLU(),
                nn.LayerNorm(64),
                nn.Linear(64, env.action_space.n)
            )
        elif mode=='Residue':
            self.net = nn.Sequential(
                ResidualBlock(env.observation_space.shape[0], 256),
                ResidualBlock(256, 256),
                ResidualBlock(256, env.action_space.n)
            )
        else:
            raise Exception('Invalid mode of network!')

    def forward(self, state):
        return F.softmax(self.net(state), dim=1)

class Critic(nn.Module):
    def __init__(self,mode=None):
        super(Critic, self).__init__()
        
        self.mode = mode
        # different types of network
        if mode==None:
            raise Exception('Must specify a network mode!')
        if mode=='Linear1':
            # states and actions enter network seperately
            self.fc1 = nn.Linear(env.observation_space.shape[0], 256)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(256 + env.action_space.n, 256)
            self.relu2 = nn.ReLU()
            self.value = nn.Linear(256, 1)
        elif mode=='Linear2':
            # states and actions enter network together
            self.fc1 = nn.Linear(env.observation_space.shape[0]+env.action_space.n, 256)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(256,256)
            self.relu2 = nn.ReLU()
            self.value = nn.Linear(256, 1)
        elif mode=='Residue1':
            # states and actions enter network seperately
            self.headfc = nn.Linear(env.observation_space.shape[0], 256)
            self.headrelu = nn.ReLU()
            self.residue = nn.Linear(256+env.action_space.n ,256)
            self.mainblock = nn.Sequential(
                ResidualBlock(256+env.action_space.n, 256),
                ResidualBlock(256, 256),
                ResidualBlock(256, 256+env.action_space.n),
            )
            self.value = nn.Linear(256+env.action_space.n, 1)
        elif mode=='Residue2':
            # states and actions enter network together
            self.headfc = nn.Linear(env.observation_space.shape[0]+env.action_space.n, 256)
            self.headrelu = nn.ReLU()
            self.residue = nn.Linear(256 ,256)
            self.mainblock = nn.Sequential(
                ResidualBlock(256, 256),
                ResidualBlock(256, 256),
                ResidualBlock(256, 256),
            )
            self.value = nn.Linear(256, 1)
        else:
            raise Exception('Invalid mode of network!')
        

    def forward(self, state, action):
        if self.mode=='Linear1':
            # states and actions enter network seperately
            out = self.fc1(state)
            out = self.relu1(out)
            out = torch.cat([out, action], dim=1)
            out = self.fc2(out)
            out = self.relu2(out)
            out = self.value(out)
        elif self.mode=='Linear2':
            # states and actions enter network together
            out = torch.cat([state, action], dim=1)
            out = self.fc1(out)
            out = self.relu1(out)
            out = self.fc2(out)
            out = self.relu2(out)
            out = self.value(out)
        elif self.mode=='Residue1':
            # states and actions enter network seperately
            out = self.headfc(state)
            out = self.headrelu(out)
            out = torch.cat([out, action], dim=1)
            identity = out
            out = self.mainblock(out)
            out = self.value(out+identity)
        elif self.mode=='Residue2':
            # states and actions enter network together
            out = torch.cat([state, action], dim=1)
            out = self.headfc(out)
            out = self.headrelu(out)
            identity = out
            out = self.mainblock(out)
            out = self.value(out+identity)
        return out
    
def get_action_distribution(state, actor):
    # print(state)
    state = torch.FloatTensor(np.array(state)).unsqueeze(0).cuda()
    action = actor(state).cpu().detach().numpy()[0]
    return action

def update_target(target_network, network, tau):
    for target_param, param in zip(target_network.parameters(), network.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def train_episodes(models:tuple, reward_list, replay_buffer, episode_index, episodes):
    
    training_log_lines = []
    
    # unpack models
    actor, critic, target_actor, target_critic = models
    # model modes
    actor_mode = actor.mode
    critic_mode = critic.mode
    
    # create env
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    #env = gym.wrappers.Monitor(env, 'videos')
    
    # copy params into the network
    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
        target_param.data.copy_(param.data)
    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.copy_(param.data)

    # optims
    actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)
    
    # begin training
    for episode in tqdm.trange(episodes):
        state,_ = env.reset()
        env.render()
        episode_reward = 0
        steps = 0
        
        frames = [] # to collect frames to generate a gif of this episode
        
        while True:
            action_distribution = get_action_distribution(state, actor)
            action = np.argmax(action_distribution)
            
            # next_state, reward, done, _ = env.step(action)
            ret = env.step(action)
            next_state, reward, done, info ,  _ = ret
            
            # frame = env.render(mode='rgb_array')
            frame = env.render()
            frames.append(frame)
            
            replay_buffer.append((state, action_distribution, reward, next_state, done))
            if len(replay_buffer) > BUFFER_SIZE:
                replay_buffer.pop(0)
            if len(replay_buffer) > BATCH_SIZE:
                batch = random.sample(replay_buffer, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.FloatTensor(np.array(states)).cuda()
                actions = torch.FloatTensor(np.array(actions)).cuda()
                rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).cuda()
                next_states = torch.FloatTensor(np.array(next_states)).cuda()
                dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).cuda()

                # training critic
                next_actions = target_actor(next_states)
                target_Q_values = target_critic(next_states, next_actions)
                y = rewards + (1 - dones) * GAMMA * target_Q_values
                Q_values = critic(states, actions)
                critic_loss = F.mse_loss(Q_values.cpu(), y.cpu().detach())
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # training actor
                actions_pred = actor(states)
                actor_loss = -critic(states, actions_pred).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # updating target network
                update_target(target_critic, critic, TAU)
                update_target(target_actor, actor, TAU)

            state = next_state
            episode_reward += reward
            
            steps += 1
            
            if done:
                
                # save params
                if episode_reward > 200:
                    torch.save(actor.state_dict(), f'./state_dicts/{actor_mode}_{critic_mode}_actor_model_{episode+episode_index}_reward_{episode_reward}.pth')
                    torch.save(critic.state_dict(), f'./state_dicts/{actor_mode}_{critic_mode}_critic_model_{episode+episode_index}_reward_{episode_reward}.pth')
                
                break
        reward_list.append(episode_reward)
        imageio.mimsave(f'./gifs/{actor_mode}_{critic_mode}_lander_episode_{episode+episode_index}.gif', frames, fps=30) # generate a gif
        print(f"{actor_mode}_{critic_mode} Episode {episode+episode_index}, Reward: {episode_reward}, Steps: {steps}")
        
        # append log to content
        training_log_lines.append(
            f"{actor_mode}_{critic_mode} Episode {episode+episode_index}, Reward: {episode_reward}, Steps: {steps}"
        )
        with open(f'./logs/{actor_mode}_{critic_mode}_log.txt','a+',encoding='utf-8') as f:
            f.write(training_log_lines[-1]+'\n')


def main():
    actor_modes = ['Linear','LinearNorm','Residue']
    critic_modes = ['Linear1', 'Linear2','Residue1','Residue2']
    
    for actor_mode in actor_modes:
        for critic_mode in critic_modes:
            
            # remove the original data of log
            f = open(f'./logs/{actor_mode}_{critic_mode}_log.txt','w',encoding='utf-8')
            f.close()
            
            # create actor and critic instance
            actor = Actor(mode=actor_mode).cuda()
            critic = Critic(mode=critic_mode).cuda()
            target_actor = Actor(mode=actor_mode).cuda()
            target_critic = Critic(mode=critic_mode).cuda()
            models = (actor, critic, target_actor, target_critic)
            
            # replay buffer
            replay_buffer = []
            # rewards for plotting
            reward_list = []
            
            # begin training
            train_episodes(models, reward_list, replay_buffer, 0, 1000)


if __name__ == '__main__':
    main()