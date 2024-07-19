import torch
from torch.utils.data import DataLoader, TensorDataset
from hmm import Mess3Process, RRXORProcess
from transformer_model import SimpleTransformer
from train import train_model
from analyze import analyze_model, compare_belief_states_and_next_token

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
