import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def parse(filename):
    with open(file=filename, mode='r', encoding='utf-8') as f:
        episodes = []
        rewards = []
        steps = []
        
        lines = f.readlines()
        for line in lines:
            parts = line.split(', ')
            episode = parts[0].split()[-1]
            reward = parts[1].split()[-1]
            step = parts[2].split()[-1]
            
            episodes.append(int(episode))
            rewards.append(float(reward))
            steps.append(int(step))
    return np.array(episodes), np.array(rewards), np.array(steps)

def subplot(ax, episodes, rewards, steps):
    steps[steps>1000] = 1000 # Filter
    ax.plot(episodes, rewards, label='rewards', color='blue', alpha=0.5)
    ax.plot(episodes, np.zeros_like(episodes), linestyle='--', color='green', label='"0" line')
    # adjust the ticks of rewards
    # ax.set_yticks(np.arange(np.min(rewards), np.max(rewards) + 0.2, (np.max(rewards) +0.2 -np.min(rewards))/10))
    ax.set_ylim([-2000,340])
    ax.set_yticks(np.arange(-320*6, 321, 320))
    
    ax2 = ax.twinx()
    ax2.plot(episodes, steps, label='steps', color='red', alpha=0.7)
    # adjust the ticks of steps
    # ax2.set_yticks(np.arange(np.min(steps), np.max(steps) + 0.2, (np.max(steps) +0.2 -np.min(steps))/10))
    ax2.set_ylim([0,2000])
    ax2.set_yticks(np.arange(0, 2001, 200))
    
    return ax, ax2

def main():
    
    actor_modes = ['Linear','LinearNorm','Residue']
    critic_modes = ['Linear1','Linear2','Residue1','Residue2']
    
    fig = plt.figure(figsize=(18,9))
    
    axs = fig.subplots(len(actor_modes), len(critic_modes))
    plt.subplots_adjust(hspace=0.4, wspace=0.7)
    ax1s, ax2s = [],[]
    
    for i,actor_mode in enumerate(actor_modes):
        for j,critic_mode in enumerate(critic_modes):
            # read log and find data
            episodes, rewards, steps = parse(f'./logs/{actor_mode}_{critic_mode}_log.txt')

            ax1, ax2 = subplot(axs[i,j], episodes, rewards, steps)
            
            ax1.set_xlabel('episodes')
            ax1.set_ylabel('rewards')
            ax2.set_ylabel('steps')
            ax1.set_title(f'Actor:{actor_mode}   Critic:{critic_mode}')
            
            ax1s.append(ax1)
            ax2s.append(ax2)
    lines1, labels1 = ax1s[0].get_legend_handles_labels()
    lines2, labels2 = ax2s[0].get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    suptitle_obj = fig.suptitle('Rewards and Steps curve')
    font = FontProperties()
    font.set_family('SimHei')
    font.set_size(24)
    font.set_weight('bold')
    suptitle_obj.set_fontproperties(font)
    
    plt.show()
        
            


if __name__ == '__main__':
    main()