import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

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
        self.T_A = self.T_A / self.T_A.sum(axis=1, keepdims=True)
        self.T_B = self.T_B / self.T_B.sum(axis=1, keepdims=True)
        self.T_C = self.T_C / self.T_C.sum(axis=1, keepdims=True)

    def generate_sequence(self, length):
        states = np.zeros(length, dtype=int)
        observations = []
        current_state = np.random.choice(self.num_states)
        for t in range(length):
            states[t] = current_state
            T_choice = np.random.choice(3)
            T = self.T_A if T_choice == 0 else self.T_B if T_choice == 1 else self.T_C
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

class SimpleTransformer(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads, num_layers):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(d_model=dim_model, nhead=num_heads, num_encoder_layers=num_layers)
        self.fc_out = nn.Linear(dim_model, num_tokens)

    def forward(self, src):
        src_emb = self.embedding(src)
        transformer_output = self.transformer(src_emb, src_emb)
        output = self.fc_out(transformer_output)
        return output

    def get_residual_stream(self, src):
        src_emb = self.embedding(src)
        return self.transformer.encoder(src_emb)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            src, tgt = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                src, tgt = batch[0].to(device), batch[1].to(device)
                output = model(src)
                loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
                val_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_transformer_model.pth')
        
        # Save model at each epoch for later analysis
        torch.save(model.state_dict(), f'transformer_model_epoch_{epoch}.pth')

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

def visualize_belief_state_geometry(residual_stream, belief_states):
    pca = PCA(n_components=3)
    residual_pca = pca.fit_transform(residual_stream)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(residual_pca[:, 0], residual_pca[:, 1], residual_pca[:, 2], c=belief_states, cmap='viridis')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Belief State Geometry in Residual Stream')
    plt.colorbar(scatter, label='Belief State')
    plt.show()

def analyze_layer_wise_representation(model, states, observations, device):
    model.eval()
    src = torch.tensor(observations[:-1], dtype=torch.long).unsqueeze(1).to(device)
    
    layer_wise_scores = []
    for layer in range(model.transformer.encoder.num_layers):
        with torch.no_grad():
            residual_stream = model.transformer.encoder.layers[:layer+1](model.embedding(src)).squeeze().cpu().numpy()
        
        belief_states = states[1:]  # Skip initial state
        X_train, X_test, y_train, y_test = train_test_split(residual_stream, belief_states, test_size=0.2, random_state=42)
        
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        layer_wise_scores.append(regressor.score(X_test, y_test))
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(layer_wise_scores) + 1), layer_wise_scores, marker='o')
    plt.xlabel("Layer")
    plt.ylabel("R² Score")
    plt.title("Layer-wise Representation of Belief States")
    plt.show()

def compare_belief_distances_across_training(model, states, observations, device, num_epochs):
    model.eval()
    src = torch.tensor(observations[:-1], dtype=torch.long).unsqueeze(1).to(device)
    
    belief_states = states[1:]  # Skip initial state
    
    distance_correlations = []
    for epoch in range(num_epochs):
        model.load_state_dict(torch.load(f'transformer_model_epoch_{epoch}.pth'))
        with torch.no_grad():
            residual_stream = model.get_residual_stream(src).squeeze().cpu().numpy()
        
        belief_distances = np.linalg.norm(residual_stream[:, np.newaxis] - residual_stream, axis=2)
        true_distances = np.abs(np.array(belief_states)[:, np.newaxis] - np.array(belief_states))
        
        correlation, _ = pearsonr(belief_distances.flatten(), true_distances.flatten())
        distance_correlations.append(correlation)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), distance_correlations)
    plt.xlabel("Training Epoch")
    plt.ylabel("Correlation between True and Predicted Belief Distances")
    plt.title("Belief Distance Correlation across Training")
    plt.show()

def analyze_entire_future_prediction(model, process, device, sequence_length=100):
    model.eval()
    states, observations = process.generate_sequence(sequence_length)
    src = torch.tensor(observations, dtype=torch.long).unsqueeze(1).to(device)
    
    with torch.no_grad():
        residual_stream = model.get_residual_stream(src).squeeze().cpu().numpy()
    
    future_predictions = []
    for t in range(sequence_length - 1):
        X = residual_stream[t]
        y = observations[t+1:]
        
        X_train, X_test, y_train, y_test = train_test_split(X.reshape(1, -1), [y], test_size=0.2, random_state=42)
        
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        future_predictions.append(regressor.score(X_test, y_test))
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(future_predictions) + 1), future_predictions)
    plt.xlabel("Time Step")
    plt.ylabel("R² Score for Future Prediction")
    plt.title("Entire Future Prediction Capability")
    plt.show()

# ... (previous code remains the same)

def examine_msp_structure(process, sequence_length=1000):
    states, observations = process.generate_sequence(sequence_length)
    
    unique_states = np.unique(states)
    state_transitions = {}
    
    for i in range(len(states) - 1):
        current_state = states[i]
        next_state = states[i+1]
        observation = observations[i]
        
        if current_state not in state_transitions:
            state_transitions[current_state] = {}
        
        if next_state not in state_transitions[current_state]:
            state_transitions[current_state][next_state] = {obs: 0 for obs in process.tokens}
        
        state_transitions[current_state][next_state][observation] += 1
    
    # Visualize MSP structure
    fig, ax = plt.subplots(figsize=(12, 8))
    for current_state in unique_states:
        for next_state in unique_states:
            if next_state in state_transitions[current_state]:
                for obs, count in state_transitions[current_state][next_state].items():
                    if count > 0:
                        ax.annotate(f"{obs}:{count}", (current_state, next_state),
                                    xytext=(5, 5), textcoords='offset points')
                        ax.arrow(current_state, current_state, next_state - current_state, 0,
                                 head_width=0.1, head_length=0.1, fc='k', ec='k')
    
    ax.set_xticks(unique_states)
    ax.set_yticks(unique_states)
    ax.set_title("Mixed State Presentation (MSP) Structure")
    ax.set_xlabel("Current State")
    ax.set_ylabel("Next State")
    plt.grid(True)
    plt.show()

def main():
    # Choose process (Mess3 or RRXOR)
    process = Mess3Process()  # or RRXORProcess()
    
    # Generate data
    states, observations = process.generate_sequence(10000)
    
    # Convert observations to integers if they're not already
    if isinstance(observations[0], str):
        token_to_int = {token: i for i, token in enumerate(process.tokens)}
        observations = [token_to_int[token] for token in observations]
    
    # Prepare data for transformer
    src = torch.tensor(observations[:-1], dtype=torch.long).unsqueeze(1)
    tgt = torch.tensor(observations[1:], dtype=torch.long).unsqueeze(1)
    dataset = TensorDataset(src, tgt)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # Model parameters
    num_tokens = len(process.tokens)
    dim_model = 64
    num_heads = 4
    num_layers = 4
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleTransformer(num_tokens, dim_model, num_heads, num_layers).to(device)
    
    # Train model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
    
    # Analyze model
    r2_score = analyze_model(model, states, observations, device)
    print(f"R2 Score: {r2_score:.4f}")
    
    # Compare belief states and next-token predictions
    compare_belief_states_and_next_token(model, states, observations, device)
    
    # Visualize belief state geometry
    with torch.no_grad():
        residual_stream = model.get_residual_stream(src.to(device)).squeeze().cpu().numpy()
    visualize_belief_state_geometry(residual_stream, states[1:])
    
    # Analyze layer-wise representation
    analyze_layer_wise_representation(model, states, observations, device)
    
    # Compare belief distances across training
    compare_belief_distances_across_training(model, states, observations, device, num_epochs)
    
    # Analyze entire future prediction capability
    analyze_entire_future_prediction(model, process, device)
    
    # Examine MSP structure
    examine_msp_structure(process)

if __name__ == "__main__":
    main()