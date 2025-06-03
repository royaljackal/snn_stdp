import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import surrogate
from snntorch import utils
import snntorch.spikeplot as splt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

# Параметры
num_steps = 100  # Временные шаги
batch_size = 16
num_inputs = 4
num_hidden = 10
num_outputs = 3
lr = 1e-2
beta = 0.95  # LIF параметр

# Загрузка и подготовка данных
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# Стандартизация признаков
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разбиение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразование в тензоры
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.Tensor(y_train)
y_test = torch.Tensor(y_test)

# Функция генерации входных спайков по частоте
def encode_inputs(x, num_steps):
    return torch.rand((num_steps, x.size(0), x.size(1))) < x.unsqueeze(0)

# Модель
class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        spk1_rec = []
        spk2_rec = []

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        for step in range(num_steps):
            cur_input = x[step]
            h1 = self.fc1(cur_input)
            spk1, mem1 = self.lif1(h1, mem1)
            h2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(h2, mem2)

            spk1_rec.append(spk1)
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec), torch.stack(spk1_rec)

# Модель, оптимизатор, функция потерь
model = SNN()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

# Обучение
def train():
    for epoch in range(10):
        permutation = torch.randperm(X_train.size(0))
        total_loss = 0
        correct = 0

        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            # Rate-coding input
            spk_in = encode_inputs(batch_x, num_steps).float()

            model.zero_grad()
            spk_out, _ = model(spk_in)

            # Суммируем спайки по времени
            out_counts = spk_out.sum(0)  # [batch, num_classes]
            target = torch.max(batch_y, 1)[1]  # в классы

            loss = loss_fn(out_counts, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = out_counts.argmax(1)
            correct += (pred == target).sum().item()

        acc = correct / X_train.size(0)
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")


# Тест
def test():
    with torch.no_grad():
        spk_in = encode_inputs(X_test, num_steps).float()
        spk_out, _ = model(spk_in)
        out_counts = spk_out.sum(0)
        pred = out_counts.argmax(1)
        target = torch.max(y_test, 1)[1]
        acc = (pred == target).float().mean().item()
        print(f"Test Accuracy: {acc:.4f}")


train()
start = time.time()
test()
end = time.time()
print(end - start)