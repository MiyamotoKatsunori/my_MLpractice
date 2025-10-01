import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. データの準備
transform = transforms.ToTensor()
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

# 2. ニューラルネットの定義
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc(x)
        return x

model = SimpleNN()

# 3. 損失関数と最適化手法
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 学習ループ（数エポック回す）
for epoch in range(4):  # 3回繰り返す
    running_loss = 0.0
    for images, labels in trainloader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}")

# 5. 正解率の計算
correct = 0
total = 0
with torch.no_grad():  # 評価時は勾配を計算しない
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # 最大のスコアを予測ラベルに
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"正解率: {100 * correct / total:.2f}%")
