import numpy as np

class HiddenMarkovModel:
    def __init__(self, transition_matrix, emission_matrix, initial_distribution):
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        self.initial_distribution = initial_distribution
        self.num_states = transition_matrix.shape[0]
        self.num_observations = emission_matrix.shape[1]

    def generate_sequence(self, length):
        states = np.zeros(length, dtype=int)
        observations = np.zeros(length, dtype=int)
        
        states[0] = np.random.choice(self.num_states, p=self.initial_distribution)
        observations[0] = np.random.choice(self.num_observations, p=self.emission_matrix[states[0]])
        
        for t in range(1, length):
            states[t] = np.random.choice(self.num_states, p=self.transition_matrix[states[t-1]])
            observations[t] = np.random.choice(self.num_observations, p=self.emission_matrix[states[t]])
        
        return states, observations

# Example HMM setup
transition_matrix = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

emission_matrix = np.array([
    [0.9, 0.1],  # Probabilities of observing [token 0, token 1] given state 0
    [0.2, 0.8]   # Probabilities of observing [token 0, token 1] given state 1
])

initial_distribution = np.array([0.5, 0.5])

if __name__ == "__main__":
    hmm = HiddenMarkovModel(transition_matrix, emission_matrix, initial_distribution)
    states, observations = hmm.generate_sequence(1000)
    np.save('states.npy', states)
    np.save('observations.npy', observations)
