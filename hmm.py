import numpy as np

class Mess3Process:
    def __init__(self):
        self.T_A = np.array([[0.765, 0.00375, 0.00375],
                             [0.0425, 0.0675, 0.00375],
                             [0.0425, 0.00375, 0.0675]])
        self.T_B = np.array([[0.0675, 0.0425, 0.00375],
                             [0.00375, 0.765, 0.00375],
                             [0.00375, 0.0425, 0.0675]])
        self.T_C = np.array([[0.0675, 0.00375, 0.0425],
                             [0.00375, 0.0675, 0.0425],
                             [0.00375, 0.00375, 0.765]])
        self.tokens = ['A', 'B', 'C']
        self.num_states = 3

    def generate_sequence(self, length):
        states = np.zeros(length, dtype=int)
        observations = []
        current_state = np.random.choice(self.num_states)
        for t in range(length):
            states[t] = current_state
            T = np.random.choice([self.T_A, self.T_B, self.T_C])
            probs = T[current_state]
            token = np.random.choice(self.tokens, p=probs)
            observations.append(token)
            current_state = np.random.choice(self.num_states, p=probs)
        return states, observations

class RRXORProcess:
    def __init__(self):
        self.T_0 = np.array([[0, 0.5, 0, 0, 0],
                             [0, 0, 0, 0, 0.5],
                             [0, 0, 0, 0.5, 0],
                             [0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0]])
        self.T_1 = np.array([[0, 0, 0.5, 0, 0],
                             [0, 0, 0, 0.5, 0],
                             [0, 0, 0, 0, 0.5],
                             [1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]])
        self.tokens = [0, 1]
        self.num_states = 5

    def generate_sequence(self, length):
        states = np.zeros(length, dtype=int)
        observations = np.zeros(length, dtype=int)
        current_state = np.random.choice(self.num_states)
        for t in range(length):
            states[t] = current_state
            T = np.random.choice([self.T_0, self.T_1])
            probs = T[current_state]
            token = np.random.choice(self.tokens, p=probs)
            observations[t] = token
            current_state = np.random.choice(self.num_states, p=probs)
        return states, observations