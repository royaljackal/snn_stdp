import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# Загрузка и подготовка данных
iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Счётчики операций
total_operations = 0
total_multiplications = 0

# Модель с учётом операций
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 10)   # Вход (4) → скрытый (10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)   # Скрытый (10) → выход (3)

    def forward(self, x):
        global total_operations, total_multiplications

        # Первый слой
        x1 = self.fc1(x)
        total_multiplications += x.size(0) * 4 * 10       # входы * выходы
        total_operations += x.size(0) * (4 * 10 + 10)      # + сдвиги (bias), + ReLU позже

        x2 = self.relu(x1)
        total_operations += x.size(0) * 10                 # ReLU: по одному на нейрон

        # Второй слой
        x3 = self.fc2(x2)
        total_multiplications += x.size(0) * 10 * 3
        total_operations += x.size(0) * (10 * 3 + 3)

        return x3

model = MLP()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Обучение
epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Отчёт по операциям
print(f"\nОбщее количество операций (сложение, умножение, ReLU): {total_operations}")
print(f"Доля операций умножения: {total_multiplications/total_operations}")
avg_ops_per_neuron = total_operations / (13 * len(X_train) * epochs)
print(f"Среднее количество операций на нейрон за один forward: {avg_ops_per_neuron:.2f}")

start = time.time()
# Тестирование
with torch.no_grad():
    y_pred = model(X_test)
    predicted = torch.argmax(y_pred, dim=1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
end = time.time()
print(end - start)