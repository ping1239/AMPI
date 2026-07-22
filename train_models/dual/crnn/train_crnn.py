import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 1. 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'dataset', 'dual')
save_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'output', 'dual', 'CNN-RNN')
os.makedirs(save_dir, exist_ok=True)

# 2. 데이터 로드 및 PyTorch DataLoader 생성
print("Loading dataset...")
X_train = np.concatenate((np.load(os.path.join(dataset_dir, 'X_orig_train.npy')), np.load(os.path.join(dataset_dir, 'X_diff_train.npy'))), axis=1)
y_train = np.load(os.path.join(dataset_dir, 'y_train.npy'))
X_val = np.concatenate((np.load(os.path.join(dataset_dir, 'X_orig_val.npy')), np.load(os.path.join(dataset_dir, 'X_diff_val.npy'))), axis=1)
y_val = np.load(os.path.join(dataset_dir, 'y_val.npy'))
X_test = np.concatenate((np.load(os.path.join(dataset_dir, 'X_orig_test.npy')), np.load(os.path.join(dataset_dir, 'X_diff_test.npy'))), axis=1)
y_test = np.load(os.path.join(dataset_dir, 'y_test.npy'))
class_names = np.load(os.path.join(dataset_dir, 'classes.npy')).tolist()

if X_train.ndim != 4 or X_train.shape[1:3] != (12, 65):
    raise ValueError(f"Expected dataset shape (N, 12, 65, window_size), got {X_train.shape}")
if len(class_names) != 5:
    raise ValueError(f"Expected 5 classes, got {class_names}")

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

# 3. CRNN (Convolutional Recurrent Neural Network) 모델 정의
class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        
        # [CNN 영역] 스펙트로그램의 공간/주파수 특징 추출
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (Batch, 32, 32, 32)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (Batch, 64, 16, 16)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 시간 축(Time, 가로)은 16을 유지하고 주파수 축(Freq, 세로)만 8로 반감
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # (Batch, 128, 8, 16)
        )
        
        # [RNN 영역] 시간 축 변화량 학습 (Bi-GRU 사용)
        # 입력 차원: 128 (채널 수) * 8 (축소된 주파수 축 크기) = 1024
        self.gru = nn.GRU(
            input_size=1024, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True
        )
        
        # [분류기 영역]
        # 양방향 GRU이므로 hidden_size * 2 크기가 입력으로 들어옴
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        # x shape: (Batch, 12, 65, 64) - 6 orig + 6 diff channels
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)  # shape: (Batch, 128, 8, 16)
        
        # RNN 입력 차원에 맞게 변경: (Batch, Channels, Freq, Time) -> (Batch, Time, Channels * Freq)
        # 1. 차원 재배치: (Batch, Time=16, Channels=128, Freq=8)
        x = x.permute(0, 3, 1, 2)
        # 2. 평탄화: (Batch, 16, 1024)
        batch_size, seq_len, _, _ = x.size()
        x = x.contiguous().view(batch_size, seq_len, -1)
        
        # GRU 통과: 출력 shape (Batch, Seq_len, Hidden_size * 2)
        gru_out, _ = self.gru(x)
        
        # 마지막 시점(Time Step)의 출력값 선택
        out = gru_out[:, -1, :]  # (Batch, 256)
        
        # 분류기 통과
        out = self.classifier(out)
        return out

class DualCRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.orig_branch = CRNN()
        self.diff_branch = CRNN()
        self.fusion = nn.Linear(10, 5)

    def forward(self, x):
        orig_output = self.orig_branch(x[:, :6])
        diff_output = self.diff_branch(x[:, 6:])
        return self.fusion(torch.cat((orig_output, diff_output), dim=1))


model = DualCRNN().to(device)

# 4. 손실 함수 및 옵티마이저
class_counts = np.bincount(y_train, minlength=5)
class_weights = len(y_train) / (5 * class_counts)
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 모델 학습 루프
num_epochs = 15
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("Starting training...")
best_val_acc = 0.0

for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_train_loss = running_loss / len(train_loader)
    epoch_train_acc = 100 * correct / total
    
    # --- Validation ---
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    epoch_val_loss = val_loss / len(val_loader)
    epoch_val_acc = 100 * correct / total
    
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
        torch.save(model.state_dict(), os.path.join(save_dir, 'crnn_best.pth'))

print("Training finished!")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

# 6. 학습 결과 그래프 저장
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss over Epochs (CRNN)')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.title('Accuracy over Epochs (CRNN)')
plt.xlabel('Epochs')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'training_history.png'))
print(f"Graph saved to {save_dir}/training_history.png")
plt.close()

# 7. 최종 평가
print("\n--- Evaluating on Test Set ---")
model.load_state_dict(torch.load(os.path.join(save_dir, 'crnn_best.pth'), weights_only=True))
model.eval()
test_correct = 0
test_total = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = 100 * test_correct / test_total
print(f"Final Test Accuracy: {test_acc:.2f}%")

cm = confusion_matrix(all_labels, all_predictions, labels=list(range(5)))
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
display.plot(cmap=plt.cm.Blues)
plt.title('CRNN Test Set Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
plt.close()
