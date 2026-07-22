import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import efficientnet_b0
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. 경로 설정 (상대 경로)
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'dataset', 'dual')
save_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'output', 'dual', 'EfficientNet')
os.makedirs(save_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. 데이터 로드
print("Loading dataset...")
X_train = np.concatenate((np.load(os.path.join(data_dir, 'X_orig_train.npy')), np.load(os.path.join(data_dir, 'X_diff_train.npy'))), axis=1)
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
X_val = np.concatenate((np.load(os.path.join(data_dir, 'X_orig_val.npy')), np.load(os.path.join(data_dir, 'X_diff_val.npy'))), axis=1)
y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
X_test = np.concatenate((np.load(os.path.join(data_dir, 'X_orig_test.npy')), np.load(os.path.join(data_dir, 'X_diff_test.npy'))), axis=1)
y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

# 3. 6채널 입력용 Custom EfficientNet-B0 정의
def get_custom_efficientnet_b0():
    # Pre-trained 가중치 없이 밑바닥부터 학습
    model = efficientnet_b0(weights=None)
    
    # 입력 이미지(65x64)가 매우 작으므로, 정보 손실을 방지하기 위해 첫 Conv2d의 stride를 1로 변경
    original_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        in_channels=6, 
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=(1, 1), # (2,2) -> (1,1) 변경
        padding=original_conv.padding,
        bias=False
    )
    
    # 출력 레이어를 1개의 logit으로 변경 (이진 분류용)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 5)
    
    return model

class DualEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.orig_branch = get_custom_efficientnet_b0()
        self.diff_branch = get_custom_efficientnet_b0()
        self.fusion = nn.Linear(10, 5)

    def forward(self, x):
        orig_output = self.orig_branch(x[:, :6])
        diff_output = self.diff_branch(x[:, 6:])
        return self.fusion(torch.cat((orig_output, diff_output), dim=1))


model = DualEfficientNet().to(device)

# 4. 손실 함수, 옵티마이저 및 스케줄러 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# 5. 모델 학습 루프
num_epochs = 20
best_val_acc = 0.0

train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("Starting training...")
for epoch in range(num_epochs):
    # Train
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_train_loss = running_loss / total
    epoch_train_acc = 100 * correct / total
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            predicted = torch.argmax(outputs, dim=1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    epoch_val_loss = val_loss / val_total
    epoch_val_acc = 100 * val_correct / val_total
    
    scheduler.step(epoch_val_acc)
    
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    train_accs.append(epoch_train_acc)
    val_accs.append(epoch_val_acc)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
          
    # 최고 성능 모델 저장
    if epoch_val_acc >= best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), os.path.join(save_dir, 'efficientnet_best.pth'))

print("Training finished!")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

# 6. 학습 결과 그래프 저장
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'training_history.png'))
print(f"Graph saved to {save_dir}/training_history.png")

# 7. Test Set 평가 및 Confusion Matrix 저장
print("\n--- Evaluating on Test Set ---")
model.load_state_dict(torch.load(os.path.join(save_dir, 'efficientnet_best.pth'), weights_only=True))
model.eval()
test_correct = 0
test_total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = 100 * test_correct / test_total
print(f"Final Test Accuracy: {test_acc:.2f}%")

# Confusion Matrix 생성 및 저장
cm = confusion_matrix(all_labels, all_preds, labels=list(range(5)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['01_', '02_', '03_', '05_', '06_'])
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title('EfficientNet Test Set Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
print(f"Confusion Matrix saved to {save_dir}/confusion_matrix.png")
