import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def analyze_model(model, states, observations, device):
    model.eval()
    src = torch.tensor(observations[:-1], dtype=torch.long).unsqueeze(1).to(device)
    
    with torch.no_grad():
        residual_stream = model.get_residual_stream(src).squeeze().cpu().numpy()
    
    belief_states = states[1:]  # Skip initial state
    X_train, X_test, y_train, y_test = train_test_split(residual_stream, belief_states, test_size=0.2, random_state=42)
    
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("True Belief States")
    plt.ylabel("Predicted Belief States")
    plt.title("True vs. Predicted Belief States")
    plt.show()
    
    return regressor.score(X_test, y_test)

def compare_belief_states_and_next_token(model, states, observations, device):
    model.eval()
    src = torch.tensor(observations[:-1], dtype=torch.long).unsqueeze(1).to(device)
    
    with torch.no_grad():
        residual_stream = model.get_residual_stream(src).squeeze().cpu().numpy()
        next_token_probs = torch.softmax(model(src), dim=-1).squeeze().cpu().numpy()
    
    belief_states = states[1:]  # Skip initial state
    
    # Compute pairwise distances for belief states
    belief_distances = np.linalg.norm(residual_stream[:, np.newaxis] - residual_stream, axis=2)
    
    # Compute pairwise distances for next-token predictions
    next_token_distances = np.linalg.norm(next_token_probs[:, np.newaxis] - next_token_probs, axis=2)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(belief_distances.flatten(), next_token_distances.flatten(), alpha=0.1)
    plt.xlabel("Belief State Distances")
    plt.ylabel("Next-Token Prediction Distances")
    plt.title("Belief States vs Next-Token Predictions")
    
    plt.subplot(1, 2, 2)
    plt.scatter(belief_states, np.argmax(next_token_probs, axis=1), alpha=0.1)
    plt.xlabel("True Belief States")
    plt.ylabel("Predicted Next Token")
    plt.title("Belief States vs Predicted Tokens")
    
    plt.tight_layout()
    plt.show()
