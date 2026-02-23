import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# --- 1. 定義大腦結構 (必須與訓練時完全一致) ---
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

# --- 2. 設定類別名稱 ---
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def predict_image(image_path):
    # 設備設定
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 載入模型
    model = Net().to(device)
    if not os.path.exists("fashion_mnist_model.pth"):
        print("❌ 找不到模型檔 fashion_mnist_model.pth，請先執行 mytorch.py 進行訓練！")
        return

    model.load_state_dict(torch.load("fashion_mnist_model.pth", map_location=device))
    model.eval()

    # 圖片預處理：轉為灰階、縮放為 28x28、轉為張量、歸一化
    transform = transforms.Compose([
        transforms.RandomInvert(p=1),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 讀取圖片
    try:
        img = Image.open(image_path)
        img_tensor = transform(img).unsqueeze(0).to(device) # 增加 Batch 維度
    except Exception as e:
        print(f"❌ 讀取圖片失敗: {e}")
        return

    # 進行預測
    with torch.no_grad():
        output = model(img_tensor)
        prediction = output.argmax(dim=1, keepdim=True).item()
        confidence = torch.exp(output).max().item() * 100

    print(f"🔍 預測結果: {classes[prediction]} (信心指數: {confidence:.2f}%)")

    # 顯示結果
    plt.imshow(img, cmap='gray')
    plt.title(f"Predict: {classes[prediction]} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # 你可以把一張衣服的照片丟進資料夾，並改下面這個檔名
    test_file = "test_item.jpg" 
    if os.path.exists(test_file):
        predict_image(test_file)
    else:
        print(f"請準備一張圖片並命名為 {test_file}，或修改程式碼中的路徑。")