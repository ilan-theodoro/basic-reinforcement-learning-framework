import numpy as np
import gym
env = gym.make("FrozenLake-v1")

from collections import namedtuple
import numpy as np

class MonteCarloControl:
    def __init__(self, num_episodes=1000):
        self.env = env
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.returns = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.π = np.zeros(self.env.observation_space.n)
        self.γ = 0.9
        self.ε = 0.1
        self.α = 0.1
        self.num_episodes = num_episodes

    def run(self):
        self.reset()
        for i in range(self.num_episodes):
            state_action_pairs = self.run_episode()
            G = self.compute_returns(state_action_pairs)
            self.update_policy(G)

        π_star = np.argmax(self.Q, axis=1)

        return self.Q, π_star

    # def update(self, state, action, reward):
    #     self.returns[state[0], state[1], action] += reward
    #     self.Q[state[0], state[1], action] = self.returns[state[0], state[1], action] / self.π[state[0], state[1], action]
    #
    def run_episode(self):
        self.env.reset()

        done = False
        state = self.env.current_position
        action = self.π[state]

        state_action_pairs = [(state, action, 0)]

        while not done:
            state, reward, done = self.env.step(action)
            if not done:
                action = self.π[state]
                state_action_pairs.append

class Agent:
   def __init__(self, scale=40, explore=10.):
      self.full_memory = {}  # format : {(state, action): [count, score]}
      self.episode_memory = []  # Record the current memory
      self.action_value = {}  # evaluate the actions by states (to discretize)
      self.experience = namedtuple("experience", ["state", "action", "reward"])
      self.scale = scale
      self.new = 0
      self.old = 0
      self.explore = explore

   # Methods

   def _discretize(self, state):
      '''discretize the state and actions'''
      state = (state * self.scale).astype(int)
      return tuple(state)

   def get_states(self, state, action, reward):
      '''
      Add the current (state, action, reward) tuple to the agent episode memory
      values are discretized
      '''
      state = self._discretize(state)
      exp = self.experience(state, action, reward)
      self.episode_memory.append(exp)

   def reset_memory(self):
      self.episode_memory = []

   def feedback_episode(self):

      # update full memory
      episode_length = len(self.episode_memory)
      for i, sa_pair in enumerate(self.episode_memory):
         state, action, _ = sa_pair
         if (state, action) in self.full_memory.keys():
            count, score = self.full_memory[(state, action)]
            score = ((count * score) + (episode_length - i)) / (count + 1)
            count += 1
            self.full_memory[(state, action)] = [count, score]
         else:
            count = 1
            score = episode_length - i
            self.full_memory[(state, action)] = [count, score]
      # Returns the number of new and old entries in memory, then reset
      proportion = (self.new, self.old)
      self.new, self.old = 0, 0
      return proportion

   def act(self, state):
      state = self._discretize(state)
      best_score = 0
      t = np.random.uniform()
      for key, value in self.full_memory.items():
         if (state == key[0]) and (best_score < value[1]):
            best_score = value[1]
            action = key[1]
      if (best_score > 0) and (t > self.explore / best_score):
         self.old += 1
         return action
      else:
         self.new += 1
         return np.random.randint(2)


agent = Agent(scale=12, explore=14.)
env = gym.make('CartPole-v1')
#agent.full_memory = save_mem
score_List = []
percent_new = []
for i_episode in range(5000):
    state, _ = env.reset()
    while True:
        # uncomment next line for graphics
        # env.render()
        action = agent.act(state) # TO DO: select action
        new_state, reward, done, _, info = env.step(action)
        # memorize step
        agent.get_states(state, action, reward)
        if done:
            new, old = agent.feedback_episode()
            score = len(agent.episode_memory)
            score_List.append(score)
            agent.reset_memory()
            percent_new.append(new*100./(new + old))
            if(len(percent_new) > 20):
                mean = np.array(percent_new)[-20:].mean()
                std = np.array(percent_new)[-20:].std()
            else:
                mean = 100.
                std = 0.
            print("Score for episode {:3d}: {:3d},    {:3.2f}% +- {:2.2f} of new moves"
                  .format(i_episode, score, mean, std),
                  end='\r')
            break
        state = new_state
env.close()
print("Last score: {}".format(score_List[-1]))