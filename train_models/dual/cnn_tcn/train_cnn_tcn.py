import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'dataset', 'dual')
save_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'output', 'dual', 'CNN-TCN')
os.makedirs(save_dir, exist_ok=True)

# 2. 데이터 로드 및 PyTorch DataLoader 생성
print("Loading dataset...")
X_train = np.concatenate((np.load(os.path.join(dataset_dir, 'X_orig_train.npy')), np.load(os.path.join(dataset_dir, 'X_diff_train.npy'))), axis=1)
y_train = np.load(os.path.join(dataset_dir, 'y_train.npy'))
X_val = np.concatenate((np.load(os.path.join(dataset_dir, 'X_orig_val.npy')), np.load(os.path.join(dataset_dir, 'X_diff_val.npy'))), axis=1)
y_val = np.load(os.path.join(dataset_dir, 'y_val.npy'))
X_test = np.concatenate((np.load(os.path.join(dataset_dir, 'X_orig_test.npy')), np.load(os.path.join(dataset_dir, 'X_diff_test.npy'))), axis=1)
y_test = np.load(os.path.join(dataset_dir, 'y_test.npy'))

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

# --- TCN (Temporal Convolutional Network) 컴포넌트 정의 ---

class Chomp1d(nn.Module):
    """
    Causal Convolution을 구현하기 위해 패딩으로 인해 늘어난 오른쪽 뒷부분을 잘라내는 레이어
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    TCN의 기본 잔차 블록 (Residual Block)
    Causal Conv1d -> BatchNorm -> ReLU -> Dropout -> Causal Conv1d -> BatchNorm -> ReLU -> Dropout
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # 첫 번째 Causal Conv Layer
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 두 번째 Causal Conv Layer
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2
        )
        
        # 잔차 연결 (Residual Connection) 시 입력 채널과 출력 채널이 다른 경우 조정을 위한 1x1 Conv
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    여러 크기의 Dilation을 가진 TemporalBlock들을 쌓아 올린 네트워크
    """
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # Dilation 크기가 2의 거듭제곱으로 증가 (1, 2, 4, 8 ...)
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # Causal 관계 유지를 위해 한쪽에 필요한 패딩 계산
            padding = (kernel_size - 1) * dilation_size
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, 
                                     dilation=dilation_size, padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# --- CNN-TCN 모델 정의 ---

class CNN_TCN(nn.Module):
    def __init__(self):
        super(CNN_TCN, self).__init__()
        
        # [CNN 영역] 스펙트로그램 특징 추출
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
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # (Batch, 128, 8, 16)
        )
        
        # [TCN 영역] 시간 축 변화 패턴 분석
        # 입력 채널: CNN에서 압축 후 추출된 차원수 (128 * 8 = 1024)
        # 시간 시퀀스 길이: 16
        # 채널 변화 구조: [128, 128] (2개 레이어 블록)
        self.tcn = TemporalConvNet(
            num_inputs=1024, 
            num_channels=[128, 128], 
            kernel_size=3, 
            dropout=0.2
        )
        
        # [분류기 영역]
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        # x shape: (Batch, 6, 65, 64)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)  # shape: (Batch, 128, 8, 16)
        
        # TCN 입력 규격에 맞게 변환: (Batch, Feature, Time)
        # 1. 차원 재배치: (Batch, Time=16, Channels=128, Freq=8)
        x = x.permute(0, 3, 1, 2)
        # 2. 평탄화 후 전치: (Batch, Time=16, 1024) -> (Batch, Feature=1024, Time=16)
        batch_size, seq_len, _, _ = x.size()
        x = x.contiguous().view(batch_size, seq_len, -1)
        x = x.transpose(1, 2)  # shape: (Batch, 1024, 16)
        
        # TCN 통과: (Batch, 128, 16)
        tcn_out = self.tcn(x)
        
        # 시간 시퀀스의 마지막 타임스텝 선택: (Batch, 128)
        out = tcn_out[:, :, -1]
        
        # 분류기 통과
        out = self.classifier(out)
        return out

class DualCNN_TCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.orig_branch = CNN_TCN()
        self.diff_branch = CNN_TCN()
        self.fusion = nn.Linear(10, 5)

    def forward(self, x):
        orig_output = self.orig_branch(x[:, :6])
        diff_output = self.diff_branch(x[:, 6:])
        return self.fusion(torch.cat((orig_output, diff_output), dim=1))


model = DualCNN_TCN().to(device)

# 4. 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
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
        torch.save(model.state_dict(), os.path.join(save_dir, 'cnn_tcn_best.pth'))

print("Training finished!")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

# 6. 학습 결과 그래프 저장
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss over Epochs (CNN-TCN)')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.title('Accuracy over Epochs (CNN-TCN)')
plt.xlabel('Epochs')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'training_history.png'))
print(f"Graph saved to {save_dir}/training_history.png")
plt.show()

# 7. 최종 평가
print("\n--- Evaluating on Test Set ---")
model.load_state_dict(torch.load(os.path.join(save_dir, 'cnn_tcn_best.pth'), weights_only=True))
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

cm = confusion_matrix(all_labels, all_predictions)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot(cmap=plt.cm.Blues)
plt.title('CNN-TCN Test Set Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
plt.close()
