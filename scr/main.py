import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. データの準備（手書き数字 MNIST）
transform = transforms.ToTensor()
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 2. ニューラルネットの定義（1層のシンプルなもの）
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(28*28, 10)  # 28x28画像を10クラスに分類

    def forward(self, x):
        x = x.view(-1, 28*28)  # 画像をベクトルに変換
        x = self.fc(x)
        return x

model = SimpleNN()

# 3. 損失関数と最適化手法
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 学習ループ（1エポックだけ）
for images, labels in trainloader:
    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("学習完了！ 最初のニューラルネットを動かしました 🚀")
