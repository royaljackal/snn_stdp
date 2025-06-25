import torch
import torch.nn as nn
import torch.optim as optim
from tensorflow.keras.datasets import mnist
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
X_test = X_test[:200]
y_test = y_test[:200]


X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 750)
        self.fc2 = nn.Linear(750, 150)
        self.fc3 = nn.Linear(150, 10)
        self.relu = nn.ReLU()

        self.total_mul_ops = 0
        self.total_add_ops = 0
        self.total_neurons = 0

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.fc1(x)
        self.total_mul_ops += batch_size * self.fc1.in_features * self.fc1.out_features
        self.total_add_ops += batch_size * (self.fc1.in_features - 1) * self.fc1.out_features
        self.total_neurons += batch_size * self.fc1.out_features
        x = self.relu(x)

        x = self.fc2(x) 
        self.total_mul_ops += batch_size * self.fc2.in_features * self.fc2.out_features
        self.total_add_ops += batch_size * (self.fc2.in_features - 1) * self.fc2.out_features
        self.total_neurons += batch_size * self.fc2.out_features
        x = self.relu(x)

        x = self.fc3(x)
        self.total_mul_ops += batch_size * self.fc3.in_features * self.fc3.out_features
        self.total_add_ops += batch_size * (self.fc3.in_features - 1) * self.fc3.out_features
        self.total_neurons += batch_size * self.fc3.out_features

        return x

    def report_ops(self):
        total_ops = self.total_mul_ops + self.total_add_ops
        mul_ratio = self.total_mul_ops / total_ops if total_ops else 0
        avg_ops_per_neuron = total_ops / self.total_neurons if self.total_neurons else 0
        print(f"Операций: {total_ops}")
        print(f"Доля умножений: {mul_ratio:.2f}")
        print(f"Операций за проход: {avg_ops_per_neuron:.2f}")

model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    outputs = model(X_test)
    preds = torch.argmax(outputs, dim=1)
    acc = (preds == y_test).float().mean().item()
    print(f"\nAccuracy: {acc:.2f}")

model.report_ops()

model_cpu = MLP().to("cpu")
model_cpu.load_state_dict(model.state_dict())
X_test_cpu = X_test.to("cpu")

with torch.no_grad():
    start = time.time()
    _ = model_cpu(X_test_cpu)
    end = time.time()
    print(f"Time elapsed: {end - start}")