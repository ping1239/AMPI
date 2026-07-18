import numpy as np
import os

ab_path = 'c:/Users/a0102/OneDrive/Desktop/STFT/ResNet18/abnormal_image/abnormal_image.npy'
norm_path = 'c:/Users/a0102/OneDrive/Desktop/STFT/ResNet18/normal_image/normal_image.npy'
save_dir = 'c:/Users/a0102/OneDrive/Desktop/STFT/ResNet18/dataset'

os.makedirs(save_dir, exist_ok=True)

def create_windows(data, window_size=64, step_size=16):
    """
    (Freq, Time, Channels) -> (Channels, Freq, Time_Window) 단위로 조각냅니다.
    """
    time_len = data.shape[1]
    windows = []
    # 데이터 길이가 창 크기보다 작으면 빈 배열 반환
    if time_len < window_size:
        return np.array([]).reshape(0, data.shape[2], data.shape[0], window_size)
        
    for start in range(0, time_len - window_size + 1, step_size):
        end = start + window_size
        window = data[:, start:end, :] # (65, 64, 6)
        window = np.transpose(window, (2, 0, 1)) # (6, 65, 64)
        windows.append(window)
    return np.array(windows)

def split_raw_timeline(data, train_ratio=0.7, val_ratio=0.15):
    """
    원본 타임라인을 지정된 비율로 완전히 분리하여 반환합니다.
    """
    time_len = data.shape[1]
    t1 = int(time_len * train_ratio)
    t2 = int(time_len * (train_ratio + val_ratio))
    
    train_data = data[:, :t1, :]
    val_data = data[:, t1:t2, :]
    test_data = data[:, t2:, :]
    return train_data, val_data, test_data

print("Loading raw tensors...")
ab_data = np.load(ab_path)
norm_data = np.load(norm_path)

# 1. 자르기 전에 타임라인부터 분리 (100% 격리)
print("Splitting raw timelines strictly...")
norm_train, norm_val, norm_test = split_raw_timeline(norm_data)
ab_train, ab_val, ab_test = split_raw_timeline(ab_data)

# 2. 각각 분리된 덩어리 안에서만 슬라이딩 윈도우 적용
def process_split(n_data, a_data, name):
    print(f"Slicing {name} set...")
    X_n = create_windows(n_data, window_size=64, step_size=16)
    X_a = create_windows(a_data, window_size=64, step_size=16)
    
    y_n = np.zeros(len(X_n), dtype=np.int64)
    y_a = np.ones(len(X_a), dtype=np.int64)
    
    if len(X_n) > 0 and len(X_a) > 0:
        X = np.concatenate((X_n, X_a), axis=0).astype(np.float32)
        y = np.concatenate((y_n, y_a), axis=0)
    elif len(X_n) > 0:
        X, y = X_n.astype(np.float32), y_n
    elif len(X_a) > 0:
        X, y = X_a.astype(np.float32), y_a
    else:
        return None, None
        
    print(f"[{name}] Total: {len(y)} (Normal: {len(y_n)}, Abnormal: {len(y_a)})")
    return X, y

X_train, y_train = process_split(norm_train, ab_train, "Train")
X_val, y_val = process_split(norm_val, ab_val, "Validation")
X_test, y_test = process_split(norm_test, ab_test, "Test")

# 3. 저장
print(f"\nDataset Preparation Complete!")
np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
np.save(os.path.join(save_dir, 'y_val.npy'), y_val)
np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
print(f"Saved strictly isolated chunks to {save_dir}/")
