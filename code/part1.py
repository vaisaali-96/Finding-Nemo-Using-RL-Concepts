import gym
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

#Defining the Finding Nemo Environment as follows
class FindingNemoEnvironment(gym.Env):

    def __init__(self, environment_type):

        self.environment_type = environment_type

        self.total_width = 4
        self.total_height = 4

        self.total_area = self.total_width * self.total_height

        self.observation_space = spaces.Discrete(self.total_area)
        self.action_space = spaces.Discrete(4)

        self.agent_pos = np.asarray([1, 0]) # Agent 
        self.nemo_pos = np.asarray([2, 3]) # Goal : 20 points
        self.dory_pos = np.asarray([[1, 1]]) # Dory Clue : 10 points
        self.turtle_pos = np.asarray([[0, 0], [0, 2], [2, 1], [3, 3]]) # Turtle Guide : 5 points
        self.net_pos = np.asarray([[0, 1], [0, 3], [3, 0]]) # Fishing Net Trap : -20 points
        self.jelly_pos = np.asarray([[1, 2], [3, 2]]) # Jelly Fish colony crossing hurdle : -10 points

        self.state_isUsed_mapping = {}
        self.resetUsedRewardMapping();
        self.state_reward_mapping = {"agent_pos":0,"nemo_pos":20,"dory_pos":10,"turtle_pos":5,"net_pos":-20,"jelly_pos":-10}

        self.state_stochastic_probability_mapping = {"dory_pos":0.6,"turtle_pos":0.5,"net_pos":0.7,"jelly_pos":0.3}

        # for i in range(self.total_height):
        #     for j in range(self.total_width):
        #         self.state_isUsed_mapping[f'{np.asarray([i, j])}'] = 0;

        self.rewards = 0;

        # Initializing termination Condition tracking variables
        self.found = 0 # Nemo found=1, not found=0
        self.timesteps = 0 # No. of actions taken by the agent

        self.max_timesteps = 30

        # Creating the mapping from the co-ordinates to the state.
        self.coordinates_state_mapping = {}
        for i in range(self.total_height):
            for j in range(self.total_width):
                self.coordinates_state_mapping[f'{np.asarray([i, j])}'] = i * self.total_width + j
        
    def resetUsedRewardMapping(self):
        for i in range(self.total_height):
            for j in range(self.total_width):
                self.state_isUsed_mapping[f'{np.asarray([i, j])}'] = 0
    
    def reset(self): # Resetting the environment when a termination condition is satisfied
        if self.found == 1: 
            print("\nGoal reached!!! Details : ")
        print("\n Reward = "+str(self.rewards)+" Timesteps = "+str(self.timesteps)+"  Resetting environment!!!!!")
        self.agent_pos = np.asarray([1, 0]) 
        observation = self.coordinates_state_mapping[f'{self.agent_pos}']
        self.timesteps = 0  # Resetting the number of steps taken by the agent.
        self.found = 0  # Resetting the Gold quantity to be 1.
        self.rewards = 0
        self.resetUsedRewardMapping()
        return observation

    def step(self, action):
        # action = agent's coordinate change
        if action == 'front':
            self.agent_pos[1] += 1
        if action == 'back':
            self.agent_pos[1] -= 1
        if action == 'right':
            self.agent_pos[0] += 1
        if action == 'left':
            self.agent_pos[0] -= 1

        # Ensuring that the agent doesn't go out of the environment.
        # if(self.agent_pos[0] not in range(0,4) or self.agent_pos[1] not in range(0,4)):
        #     pyautogui.alert("Invalid action. You care trying to move outside the boundary")
        #     # win32api.MessageBox(0, 'Invalid action. You care trying to move outside the boundary', 'ALERT')
        #     reset(self)
        #     return
        if(self.agent_pos[0] not in range(0,4) or self.agent_pos[1] not in range(0,4)):
            print("\n Invalid action. You care trying to move outside the boundary")
            self.agent_pos[0] = np.clip(self.agent_pos[0],0,3)
            self.agent_pos[1] = np.clip(self.agent_pos[1],0,3)

        self.timesteps += 1

        reward = 0

        if np.array_equal(self.agent_pos, self.nemo_pos) and self.found < 1:
            self.found = 1
            reward = 50

        for i in range(len(self.jelly_pos)): 
                if np.array_equal(self.agent_pos, self.jelly_pos[i]):
                    reward = -10
                    if self.environment_type == "Stochastic":
                        reward = 0.3*-10

        for i in range(len(self.net_pos)):  
            if np.array_equal(self.agent_pos, self.net_pos[i]):
                reward = -20
                if self.environment_type == "Stochastic":
                        reward = 0.7*-20

        if(self.state_isUsed_mapping[f'{self.agent_pos}']==0):
            if np.array_equal(self.agent_pos, self.dory_pos):
                reward = 10
                if self.environment_type == "Stochastic":
                        reward = 0.6*10

            for i in range(len(self.turtle_pos)):  
                if np.array_equal(self.agent_pos, self.turtle_pos[i]):
                    reward = 5
                    if self.environment_type == "Stochastic":
                        reward = 0.5*5


        self.state_isUsed_mapping[f'{self.agent_pos}'] = 1

        self.rewards += reward

        if self.found == 1:
            done = True
        else:
            done = False

        for i in range(len(self.net_pos)):
            if np.array_equal(self.agent_pos, self.net_pos[i]):
                done = True

        if self.timesteps == self.max_timesteps:
            done = True

        if done == True:
            self.reset()
            self.render()
        else:
            self.render()

        return "Step: "+str(self.timesteps)+" Action: "+action+" Position of agent: "+str(self.coordinates_state_mapping[f'{self.agent_pos}'])+" Reward received: "+str(reward)+" Total Reward: "+str(self.rewards)+" Resetting?: "+str(done)


    def render(self):
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        ax.set_facecolor('#87CEEB')

        def plot_image(plot_pos):
            plot_agent, plot_nemo, plot_dory, plot_turtle, plot_jelly, plot_net = \
                False, False, False, False, False, False

            if np.array_equal(self.agent_pos, plot_pos):
                plot_agent = True
            if self.found < 1:
                if np.array_equal(plot_pos, self.nemo_pos):
                    plot_nemo = True

            if(self.state_isUsed_mapping[f'{plot_pos}']==0):
                if np.array_equal(plot_pos, self.dory_pos):
                    plot_dory = True
                if any(np.array_equal(self.net_pos[i], plot_pos) for i in range(len(self.net_pos))):
                    plot_net = True
                if any(np.array_equal(self.jelly_pos[i], plot_pos) for i in range(len(self.jelly_pos))):
                    plot_jelly = True
                if any(np.array_equal(self.turtle_pos[i], plot_pos) for i in range(len(self.turtle_pos))):
                    plot_turtle = True

            if plot_agent:
                agent = AnnotationBbox(OffsetImage(plt.imread('./images/Marlin.png'), zoom=0.18),
                                       np.add(plot_pos, [0.5, 0.5]), bboxprops =dict(edgecolor='green',facecolor="#87CEEB"))
                ax.add_artist(agent)

            elif plot_nemo:
                nemo = AnnotationBbox(OffsetImage(plt.imread('./images/Nemo.png'), zoom=0.13),
                                        np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(nemo)

            elif plot_dory:
                dory = AnnotationBbox(OffsetImage(plt.imread('./images/Dory.jpeg'), zoom=0.17),
                                      np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(dory)

            elif plot_turtle:
                turtle = AnnotationBbox(OffsetImage(plt.imread('./images/Turtles.jpeg'), zoom=0.17),
                                     np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(turtle)

            elif plot_jelly:
                jelly = AnnotationBbox(OffsetImage(plt.imread('./images/Jellyfish.jpeg'), zoom=0.17),
                                        np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(jelly)

            elif plot_net:
                net = AnnotationBbox(OffsetImage(plt.imread('./images/Fishnet1.jpeg'), zoom=0.17),
                                        np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(net)


        for j in range(self.total_height * self.total_width):
            plot_image(np.asarray(
                [int(np.floor(j / self.total_width)), j % self.total_width]))

        plt.xticks([0, 1, 2, 3])
        plt.yticks([0, 1, 2, 3])
        plt.grid()
        plt.show()
        plt.pause(0.05)
        plt.close()
        return plt


print("\n\nDeterministic Environment testing\n")
nemo_world = FindingNemoEnvironment(environment_type='deterministic')
nemo_world.reset()
nemo_world.render()

action_sequence = ["front","right","right","front","front","left","front","left","back","back","front","front","front","right"]
for i in range(len(action_sequence)):
    print(nemo_world.step(action_sequence[i]))


nemo_world = FindingNemoEnvironment(environment_type='Stochastic')
print("\n\nStochastic Environment testing\n")
nemo_world.reset()
nemo_world.render()

action_sequence = ["front","right","right","front","front","left","front","left","back","back","front","front","front","right"]
for i in range(len(action_sequence)):
    print(nemo_world.step(action_sequence[i]))

