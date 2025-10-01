import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# データセット（MNIST）
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# CNNモデル定義
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 入力1ch → 出力32ch
        self.pool = nn.MaxPool2d(2, 2)                           # 2x2 プーリング
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 32ch → 64ch
        self.fc1 = nn.Linear(64 * 7 * 7, 128)                    # 全結合層
        self.fc2 = nn.Linear(128, 10)                            # 出力10クラス

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Conv1 → ReLU → Pool
        x = self.pool(torch.relu(self.conv2(x)))  # Conv2 → ReLU → Pool
        x = x.view(-1, 64 * 7 * 7)                # 平坦化
        x = torch.relu(self.fc1(x))               # FC1 → ReLU
        x = self.fc2(x)                           # FC2（分類）
        return x

# モデル・損失関数・最適化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習ループ
for epoch in range(4):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# テストデータで精度を計算
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
