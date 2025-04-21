import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Создаем простую нейронную сеть
class NNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(NNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, out_size),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.layers(X)

# Загружаем данные
df = pd.read_csv('dataset_simple.csv')

# Нормализуем данные
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df.iloc[:, 0:2].values)
X = torch.Tensor(X_scaled)
y = torch.Tensor(df.iloc[:, 2].values).reshape(-1, 1)

# Параметры сети
inputSize = X.shape[1]
hiddenSizes = 10  # Увеличиваем количество нейронов
outputSize = 1

# Создаем экземпляр сети
net = NNet(inputSize, hiddenSizes, outputSize)



# Ошибка до обучения
with torch.no_grad():
    pred = net(X)
pred = torch.Tensor(np.where(pred >= 0.5, 1, 0).reshape(-1, 1))
err = (y != pred).sum().item()
print(f"Ошибка до обучения: {err}")

# Функция потерь и оптимизатор
lossFn = nn.BCELoss() #BCELoss (Binary Cross Entropy Loss) — функция потерь, используемая для задач бинарной классификации
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)  # Увеличиваем скорость обучения

# Обучение
epochs = 1000
for i in range(epochs):
    pred = net(X)
    loss = lossFn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(f"Ошибка на {i + 1} итерации: {loss.item()}")

# Ошибка после обучения
with torch.no_grad():
    pred = net(X)
pred = torch.Tensor(np.where(pred >= 0.5, 1, 0).reshape(-1, 1))
err = (y != pred).sum().item()
print(f"\nОшибка после обучения: {err}")