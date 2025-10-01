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

# 2. ニューラルネットの定義（多層化！）
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)   # 入力 → 隠れ層1
        self.fc2 = nn.Linear(128, 64)      # 隠れ層1 → 隠れ層2
        self.fc3 = nn.Linear(64, 10)       # 隠れ層2 → 出力

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))        # 活性化関数 ReLU
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)                    # 出力層はそのまま
        return x

model = MLP()

# 3. 損失関数と最適化手法
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 学習ループ
for epoch in range(5): 
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
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"正解率: {100 * correct / total:.2f}%")
