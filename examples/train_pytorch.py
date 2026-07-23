import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the exact same Deep MLP architecture
class DeepMLP(nn.Module):
    def __init__(self):
        super(DeepMLP, self).__init__()
        self.fc1 = nn.Linear(784, 2048)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2048, 1024)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    # Same data transformations: PNG -> Grayscale
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    dataset_path = '/home/marwan/.dev/vecore/datasets/mnist_png/training'
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        exit(1)

    print("Loading dataset...")
    raw_dataset = datasets.ImageFolder(dataset_path, transform=transform)

    # Buffer everything into RAM Tensors like vecore
    all_inputs = []
    all_labels = []
    # Use a loader just to parallelize disk IO quickly
    disk_loader = DataLoader(raw_dataset, batch_size=2048, shuffle=False, num_workers=4)
    for inputs, labels in disk_loader:
        inputs = inputs.view(inputs.size(0), -1)
        all_inputs.append(inputs)
        all_labels.append(labels)
    
    # Concatenate into massive in-memory Tensors
    X_train = torch.cat(all_inputs, dim=0)
    Y_train = torch.cat(all_labels, dim=0)
    
    # Create an in-memory TensorDataset
    from torch.utils.data import TensorDataset
    train_dataset = TensorDataset(X_train, Y_train)

    print(f"Done! Loaded {X_train.size(0)} images into RAM.")

    # We use the same batch size of 8192 and enable pin_memory to mimic our PCIe pipelining
    batch_size = 8192
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = DeepMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    print(f"\nTraining Deep MLP on PyTorch for 150 epochs with batch size {batch_size}...")
    start_time = time.time()

    for epoch in range(150):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            # non_blocking=True is the PyTorch equivalent of our asynchronous cudaMemcpyAsync pipeline!
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        if epoch % 10 == 0 or epoch == 149:
            print(f"Epoch {epoch}/150 - Loss: {epoch_loss/total:.6f} - Acc: {100.0 * correct/total:.4f}%")

    end_time = time.time()
    print(f"\nTotal Training Time: {end_time - start_time:.2f} seconds")
    print(f"Average Time per Epoch: {(end_time - start_time) / 150:.4f} seconds")

    print("\nEvaluating on Unseen Test Images...")
    test_dataset_path = '/home/marwan/.dev/vecore/datasets/mnist_png/testing'
    
    if os.path.exists(test_dataset_path):
        test_raw_dataset = datasets.ImageFolder(test_dataset_path, transform=transform)
        test_loader = DataLoader(test_raw_dataset, batch_size=8192, shuffle=False, num_workers=4)
        
        test_correct = 0
        test_total = 0
        model.eval()
        
        eval_start_time = time.time()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                inputs = inputs.view(inputs.size(0), -1)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        eval_end_time = time.time()
        test_accuracy = test_correct / test_total * 100.0
        
        print("=====================================================")
        print(f" REAL-WORLD TEST ACCURACY: {test_accuracy:.2f}% ({test_correct}/{test_total})")
        print(f" Evaluation Time: {eval_end_time - eval_start_time:.2f} seconds")
        print("=====================================================")
    else:
        print("Test dataset not found.")
