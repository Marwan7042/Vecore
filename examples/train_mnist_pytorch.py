import torch
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
import numpy as np
import time

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        # Matching the architecture exactly: 784 -> 2048 -> 1024 -> 10
        self.layer1 = nn.Linear(784, 2048)
        self.layer2 = nn.Linear(2048, 1024)
        self.layer3 = nn.Linear(1024, 10)
        self.relu = nn.ReLU()
        
        # Match vecore's Xavier uniform init and zero bias
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def load_image(filepath, label):
    img = Image.open(filepath).convert('L')
    img = np.array(img, dtype=np.float32) / 255.0
    img = img.reshape(1, 784)
    return torch.tensor(img), torch.tensor(label, dtype=torch.long)

def main():
    print("Scanning MNIST PNG Directory...")
    
    dataset = []
    # Using path relative to this script or absolute. Assuming running from project root or examples/
    # Let's try to resolve absolute path dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, "../datasets/mnist_png/training/")
    
    # Load 100 images of each digit (1000 total) to train efficiently with batch size 1
    for label in range(10):
        dir_path = os.path.join(base_dir, str(label))
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            continue
        count = 0
        for entry in os.listdir(dir_path):
            if count >= 100:
                break
            filepath = os.path.join(dir_path, entry)
            if os.path.isfile(filepath):
                dataset.append(load_image(filepath, label))
                count += 1
                
    print(f"Loaded {len(dataset)} actual PNG images into Tensors!")
    print("Building Deep MLP (784 -> 2048 -> 1024 -> 10) to learn Computer Vision...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Transferring Weights to {device}...")
    
    model = MNISTModel().to(device)
    
    learning_rate = 0.1
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    history_loss = []
    history_acc = []
    
    start_time = time.time()
    
    for epoch in range(151):
        epoch_loss = 0.0
        correct = 0
        
        for i in range(len(dataset)):
            # 1. Forward Pass
            X_cuda, Y_target = dataset[i]
            X_cuda = X_cuda.to(device)
            Y_target = Y_target.unsqueeze(0).to(device) # batch size 1
            
            Y_pred = model(X_cuda)
            
            # 2. Calculate Loss
            loss = criterion(Y_pred, Y_target)
            
            # Check if network guessed right
            max_idx = torch.argmax(Y_pred, dim=1).item()
            if max_idx == Y_target.item():
                correct += 1
                
            epoch_loss += loss.item()
            
            # 3. Zero Gradients
            optimizer.zero_grad()
            
            # 4. Backward Pass
            loss.backward()
            
            # 5. Optimizer Step
            optimizer.step()
            
        avg_loss = epoch_loss / len(dataset)
        accuracy = correct / len(dataset) * 100.0
        
        history_loss.append(avg_loss)
        history_acc.append(accuracy)
        
        print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Accuracy: {accuracy:.2f}%")
        
    end_time = time.time()
    print(f"\nTraining Time: {end_time - start_time:.2f} seconds")
    
    # Evaluate on Test Set
    print("\nEvaluating on Unseen Test Images...")
    test_dataset = []
    test_dir = os.path.join(current_dir, "../datasets/mnist_png/testing/")
    
    for label in range(10):
        dir_path = os.path.join(test_dir, str(label))
        if not os.path.exists(dir_path):
            continue
        for entry in os.listdir(dir_path):
            filepath = os.path.join(dir_path, entry)
            if os.path.isfile(filepath):
                test_dataset.append(load_image(filepath, label))
                
    test_correct = 0
    # Add eval mode to disable dropout etc if we had any, though we don't. Good practice anyway.
    model.eval()
    
    eval_start_time = time.time()
    for i in range(len(test_dataset)):
        X_cuda, Y_target = test_dataset[i]
        X_cuda = X_cuda.to(device)
        Y_target = Y_target.unsqueeze(0).to(device)
        
        with torch.no_grad():
            Y_pred = model(X_cuda)
            max_idx = torch.argmax(Y_pred, dim=1).item()
            if max_idx == Y_target.item():
                test_correct += 1
    eval_end_time = time.time()
                
    test_accuracy = test_correct / len(test_dataset) * 100.0
    print("=====================================================")
    print(f" REAL-WORLD TEST ACCURACY: {test_accuracy:.2f}% ({test_correct}/{len(test_dataset)})")
    print(f" Evaluation Time: {eval_end_time - eval_start_time:.2f} seconds")
    print("=====================================================")

if __name__ == '__main__':
    main()
