import torch
import torch.nn as nn
import torch.optim as optim
import time

epochs = 10000
# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Benchmarking function
def benchmark(device, input_size=784, hidden_size=128, output_size=10, num_batches=100, batch_size=64, epochs=epochs):
    model = SimpleNN(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Dummy data
    inputs = torch.randn(batch_size, input_size).to(device)
    targets = torch.randint(0, output_size, (batch_size,)).to(device)

    # Warm-up (ensure accurate timing)
    for _ in range(10):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Start timing
    start_time = time.time()

    # Training loop
    for epoch in range(epochs):
        for _ in range(num_batches):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # End timing
    end_time = time.time()

    avg_time_per_epoch = (end_time - start_time) / epochs
    return avg_time_per_epoch


if __name__ == "__main__":
    input_size = 784  # Example for flattened 28x28 image (like MNIST)
    hidden_size = 128
    output_size = 10  # Example for 10 classes (e.g., MNIST digits)
    num_batches = 100
    batch_size = 64
    epochs = 10

    # CPU Benchmark
    print("Benchmarking on CPU...")
    cpu_time = benchmark(device="cpu", input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                         num_batches=num_batches, batch_size=batch_size, epochs=epochs)
    print(f"Average time per epoch on CPU: {cpu_time:.4f} seconds")

    # MPS Benchmark (if available)
    if torch.has_mps:
        print("Benchmarking on MPS...")
        mps_time = benchmark(device="mps", input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                             num_batches=num_batches, batch_size=batch_size, epochs=epochs)
        print(f"Average time per epoch on MPS: {mps_time:.4f} seconds")
    else:
        print("MPS is not supported on this device.")