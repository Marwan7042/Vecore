import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        # Matching the architecture exactly: 784 -> 2048 -> 1024 -> 512 -> 10
        self.layer1 = nn.Linear(784, 2048)
        self.layer2 = nn.Linear(2048, 1024)
        self.layer3 = nn.Linear(1024, 512)
        self.layer4 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        
        # Match vecore's Xavier uniform init and zero bias
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x

def main():
    print("Loading MNIST Dataset using torchvision...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    
    # Download dataset if not present
    train_dataset_raw = torchvision.datasets.MNIST(root='./datasets', train=True, download=True, transform=transform)
    test_dataset_raw = torchvision.datasets.MNIST(root='./datasets', train=False, download=True, transform=transform)
    
    # We will load the entire dataset into GPU memory for maximum speed, 
    # mirroring vecore's `use_pipeline=false` or fast pinned memory transfers.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Transferring Weights to {device}...")
    
    train_loader_unbatched = DataLoader(train_dataset_raw, batch_size=len(train_dataset_raw), shuffle=False)
    X_train_all, Y_train_all = next(iter(train_loader_unbatched))
    
    X_train_all = X_train_all.to(device)
    Y_train_all = Y_train_all.to(device)
    
    train_dataset = TensorDataset(X_train_all, Y_train_all)
    batch_size = 8192
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    model = MNISTModel().to(device)
    
    learning_rate = 0.3
    epochs = 150
    patience = 5
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    print("\nTraining...")
    start_time = time.time()
    
    best_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for X_batch, Y_batch in train_loader:
            # 1. Forward Pass
            Y_pred = model(X_batch)
            
            # 2. Calculate Loss
            loss = criterion(Y_pred, Y_batch)
            
            # 3. Zero Gradients
            optimizer.zero_grad()
            
            # 4. Backward Pass
            loss.backward()
            
            # 5. Optimizer Step
            optimizer.step()
            
            epoch_loss += loss.item() * X_batch.size(0)
            
            max_idx = torch.argmax(Y_pred, dim=1)
            correct += (max_idx == Y_batch).sum().item()
            total += X_batch.size(0)
            
        avg_loss = epoch_loss / total
        accuracy = (correct / total) * 100.0
        
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.6f} | Accuracy: {accuracy:.2f}%")
        
        # Early stopping logic (matching vecore's implementation)
        min_delta = 1e-4
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Loss hasn't improved by {min_delta} for {patience} epochs.")
            break
            
    end_time = time.time()
    print(f"\n>>> TOTAL TRAINING TIME: {end_time - start_time:.4f} seconds <<<")
    
    # Evaluate on Test Set
    print("\nEvaluating on Unseen Test Images...")
    
    test_loader_unbatched = DataLoader(test_dataset_raw, batch_size=len(test_dataset_raw), shuffle=False)
    X_test_all, Y_test_all = next(iter(test_loader_unbatched))
    
    X_test_all = X_test_all.to(device)
    Y_test_all = Y_test_all.to(device)
    
    model.eval()
    with torch.no_grad():
        Y_pred = model(X_test_all)
        max_idx = torch.argmax(Y_pred, dim=1)
        test_correct = (max_idx == Y_test_all).sum().item()
        
    test_accuracy = (test_correct / len(test_dataset_raw)) * 100.0
    print("=====================================================")
    print(f" REAL-WORLD TEST ACCURACY: {test_accuracy:.2f}% ({test_correct}/{len(test_dataset_raw)})")
    print("=====================================================")

if __name__ == "__main__":
    main()
