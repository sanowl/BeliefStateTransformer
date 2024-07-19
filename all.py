import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Define the Mess3Process class
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

# Define the RRXORProcess class
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

# Define the SimpleTransformer model
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

# Define the training function
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

# Define the analysis function
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

def main():
    # Choose process (Mess3 or RRXOR)
    process = Mess3Process()  # or RRXORProcess()
    
    # Generate data
    states, observations = process.generate_sequence(10000)
    
    # Convert observations to integers if they're not already
    if isinstance(observations[0], str):
        token_to_int = {token: i for i, token in enumerate(process.tokens)}
        observations = list(map(lambda token: token_to_int[token], observations))
    
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
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SimpleTransformer(num_tokens, dim_model, num_heads, num_layers).to(device)
    
    # Train model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, device=device)
    
    # Analyze model
    r2_score = analyze_model(model, states, observations, device)
    print(f"R2 Score: {r2_score:.4f}")
    
    # Compare belief states and next-token predictions
    compare_belief_states_and_next_token(model, states, observations, device)

if __name__ == "__main__":
    main()
