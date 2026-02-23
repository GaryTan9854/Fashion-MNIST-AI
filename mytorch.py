import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# --- 1. 設定與設備 ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f" usando {device} 進行運算")

# 定義 Fashion-MNIST 的 10 個類別名稱
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# --- 2. 定義大腦結構 (CNN 卷積神經網路) ---
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

# --- 3. 準備數據 (改為 Fashion-MNIST) ---
# Fashion-MNIST 的平均值與標準差略有不同，但使用 (0.5, 0.5) 也很通用
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])

# 載入訓練集
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data', train=True, download=True, transform=transform), 
    batch_size=64, shuffle=True)

model = Net().to(device)
MODEL_PATH = "fashion_mnist_model.pth"

# --- 4. 核心邏輯：讀取或訓練 ---
choice = "n" 
if os.path.exists(MODEL_PATH):
    # 這裡加入一個 try-except 防止輸入被跳過
    try:
        choice = input(f"偵測到 '{MODEL_PATH}'，是否載入？ (y/n): ").lower()
    except EOFError:
        choice = "n"

if choice == 'y':
    print("⏳ 正在載入時尚大腦...")
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("✅ 載入成功！")
else:
    print("\n🚀 開始訓練時尚分析師 (預計 5 輪，因為衣服比數字難分辨)...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(1, 6): # 增加到 5 輪效果更好
        model.train()
        start_time = time.time()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        duration = time.time() - start_time
        print(f"第 {epoch} 輪完成，平均 Loss: {running_loss/len(train_loader):.4f}，耗時: {duration:.2f} 秒")
    
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"💾 大腦已存檔至 {MODEL_PATH}")

# --- 5. 隨堂測驗 ---
print("\n--- 時尚大挑戰 ---")
model.eval()
test_data, test_target = next(iter(train_loader))
img = test_data[0].unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img)
    prediction_idx = output.argmax(dim=1, keepdim=True).item()
    prediction_name = classes[prediction_idx]
    actual_name = classes[test_target[0].item()]

print(f"AI 判斷這件衣服是: {prediction_name}")
print(f"實際答案是: {actual_name}")

# --- 6. 視覺化顯示 ---
plt.imshow(test_data[0].squeeze().cpu().numpy(), cmap='gray')
plt.title(f"AI Predict: {prediction_name}\nActual: {actual_name}")
plt.axis('off')
plt.show()