import numpy as np

class Mess3Process:
    def __init__(self):
        self.T_A = np.array([[0.765, 0.1175, 0.1175],
                             [0.915, 0.0675, 0.0175],
                             [0.91, 0.0025, 0.0875]])
        self.T_B = np.array([[0.45, 0.45, 0.1],
                             [0.1, 0.8, 0.1],
                             [0.1, 0.1, 0.8]])
        self.T_C = np.array([[0.45, 0.1, 0.45],
                             [0.1, 0.45, 0.45],
                             [0.1, 0.1, 0.8]])
        self.tokens = ['A', 'B', 'C']
        self.num_states = 3

        # Ensure each row of the transition matrices sums to 1
        self.T_A = self.T_A / self.T_A.sum(axis=1, keepdims=True)
        self.T_B = self.T_B / self.T_B.sum(axis=1, keepdims=True)
        self.T_C = self.T_C / self.T_C.sum(axis=1, keepdims=True)

    def generate_sequence(self, length):
        states = np.zeros(length, dtype=int)
        observations = []
        current_state = np.random.choice(self.num_states)
        for t in range(length):
            states[t] = current_state
            T_choice = np.random.choice(3)  # Choose 0, 1, or 2 randomly
            if T_choice == 0:
                T = self.T_A
            elif T_choice == 1:
                T = self.T_B
            else:
                T = self.T_C
            probs = T[current_state]
            token = np.random.choice(self.tokens, p=probs)
            observations.append(token)
            current_state = np.random.choice(self.num_states, p=probs)
        return states, observations

class RRXORProcess:
    def __init__(self):
        self.T_0 = np.array([[0, 0.5, 0.5, 0, 0],
                             [0, 0, 0, 0, 0.5],
                             [0, 0, 0, 0.5, 0],
                             [0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0]])
        self.T_1 = np.array([[0, 0, 0.5, 0, 0.5],
                             [0, 0, 0, 0.5, 0.5],
                             [0, 0, 0, 0, 0.5],
                             [1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]])
        self.tokens = [0, 1]
        self.num_states = 5

        # Ensure each row of the transition matrices sums to 1
        self.T_0 = self.T_0 / self.T_0.sum(axis=1, keepdims=True)
        self.T_1 = self.T_1 / self.T_1.sum(axis=1, keepdims=True)

    def generate_sequence(self, length):
        states = np.zeros(length, dtype=int)
        observations = np.zeros(length, dtype=int)
        current_state = np.random.choice(self.num_states)
        for t in range(length):
            states[t] = current_state
            T = self.T_0 if np.random.random() < 0.5 else self.T_1
            probs = T[current_state]
            token = np.random.choice(self.tokens, p=probs)
            observations[t] = token
            current_state = np.random.choice(self.num_states, p=probs)
        return states, observations
