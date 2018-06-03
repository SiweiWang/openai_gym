import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, epsilon=0.001, alpha=0.1, gamma=0.99):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        policy_s = self.epsilon_greedy_prob(state, i_episode)
        return np.random.choice(np.arange(self.nA), p=policy_s)
  
    def step(self, state, i_episode, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        # q learning
        policy_s = self.epsilon_greedy_prob(next_state, i_episode)
        self.Q[state][action] = self.update_Q(self.Q[state][action], np.max(self.Q[next_state]), reward)

        # expected sars
#         policy_s = self.epsilon_greedy_prob(next_state, i_episode, esp=self.epsilon)
#         self.Q[state][action] = self.update_Q(self.Q[state][action], np.dot(self.Q[next_state], policy_s), reward)
        
    def epsilon_greedy_prob(self, state, i_episode, esp=None):
        """ update policy using epsilon greedy method """
        epsilon = 1.0/i_episode
        if esp is not None:
            epsilon = esp
        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(self.Q[state])] = 1 - epsilon + epsilon /self.nA
        return policy_s

    def update_Q(self, Qsa, Qsa_next, reward):
        """ updates the action-value function estimate using the most recent step"""
        Qsa = Qsa + (self.alpha * (reward + (self.gamma * Qsa_next) - Qsa))
        return Qsa