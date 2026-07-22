# AMPI: STFT 기반 모터 상태 5클래스 분류

모터 전류 신호를 STFT 이미지로 변환한 뒤, `orig`와 `diff` 특징을 함께 사용하는
CNN-BiGRU(CRNN) 모델로 5개 상태를 분류하는 프로젝트입니다.

대용량 데이터셋과 학습 결과는 Git 저장소에 포함하지 않습니다. 아래 Google Drive에서
STFT 데이터를 내려받은 뒤 프로젝트 루트에 배치해야 합니다.

## 데이터 다운로드

[Google Drive에서 STFT 데이터 다운로드](https://drive.google.com/drive/folders/1qR01lLkU2tdX-h3v-imlAWxaQlpUzrGB?usp=drive_link)

Google Drive에서 데이터를 다운로드하고 압축을 푼 뒤, 다음과 같이 프로젝트 루트에
`stft_image/` 폴더가 오도록 배치합니다.

```text
AMPI/
├── dataset_prep.py
├── requirements.txt
├── train_models/
│   └── train_crnn.py
└── stft_image/
    ├── 01_/
    │   ├── session_1_orig.npy
    │   └── session_1_diff.npy
    ├── 02_/
    │   └── session_*_{orig,diff}.npy
    ├── 03_/
    │   └── session_*_{orig,diff}.npy
    ├── 05_/
    │   └── session_*_{orig,diff}.npy
    └── 06_/
        └── session_*_{orig,diff}.npy
```

`stft_image/stft_image/...`처럼 폴더가 두 번 중첩되지 않도록 주의하세요.

## 분류 클래스

클래스 폴더는 이름순으로 다음 label에 대응합니다.

| Label | 클래스 |
|---:|---|
| 0 | `01_` |
| 1 | `02_` |
| 2 | `03_` |
| 3 | `05_` |
| 4 | `06_` |

## 실행 환경 준비

Python 3.10 이상을 권장합니다.

### Windows PowerShell

```powershell
cd C:\path\to\AMPI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

기존 `.venv`가 있다면 생성 명령은 생략하고 활성화만 하면 됩니다.

## 1. 데이터셋 생성

```powershell
python dataset_prep.py
```

각 세션의 `orig.npy`와 `diff.npy`를 시간 기준으로 정렬하고 채널 방향으로 결합합니다.
각 파일이 6채널이므로 모델 입력은 총 12채널입니다.

```text
orig:     (6, 65, time)
diff:     (6, 65, time)
combined: (12, 65, time)
```

각 세션의 timeline을 먼저 train 70%, validation 15%, test 15%로 분리한 뒤 각 구간에서
window를 생성합니다. 이 순서는 겹치는 window가 서로 다른 split에 들어가는 것을
방지합니다. 정규화 통계는 train 데이터에서만 계산합니다.

생성 결과:

```text
dataset/
├── X_train.npy
├── y_train.npy
├── X_val.npy
├── y_val.npy
├── X_test.npy
├── y_test.npy
├── classes.npy
└── normalization.npz
```

`dataset/`은 자동 생성되며 Git에는 포함되지 않습니다.

## 2. CRNN 학습 및 평가

```powershell
python train_models/train_crnn.py
```

현재 5클래스·12채널 데이터셋에 맞게 구성된 학습 파일은 `train_crnn.py`입니다.
CRNN은 CNN으로 주파수 특징을 추출하고 양방향 GRU로 시간 변화를 학습합니다.

기본 학습 설정:

- Epoch: 15
- Batch size: 16
- Optimizer: Adam
- Loss: 클래스 불균형 가중치가 적용된 Cross-Entropy
- 입력: `(batch, 12, 65, window_size)`
- 출력: `(batch, 5)`

학습이 끝나면 validation accuracy가 가장 높았던 체크포인트로 test dataset을 평가합니다.

결과는 다음 경로에 저장됩니다.

```text
model_output/CNN-RNN/
├── crnn_best.pth
├── crnn_training_history.png
├── confusion_matrix.png
└── classification_report.txt
```

`model_output/`도 Git에는 포함되지 않습니다.

## Window와 Step 크기 변경

`dataset_prep.py` 상단에서 변경할 수 있습니다.

```python
WINDOW_SIZE = 128
STEP_SIZE = 32
```

현재 STFT 설정의 sampling rate는 64,000Hz이고 hop length는 1,024 sample이므로
STFT 프레임 간격은 약 0.016초입니다.

현재 설정은 대략 다음 시간에 해당합니다.

- Window 128: 약 2.048초
- Step 32: 약 0.512초

설정을 변경한 후에는 반드시 데이터셋을 다시 생성해야 합니다.

```powershell
python dataset_prep.py
python train_models/train_crnn.py
```

CRNN은 시간축 길이를 자동으로 처리하므로 `WINDOW_SIZE`를 변경할 때 학습 파일을 별도로
수정할 필요는 없습니다.

## 전체 실행 순서

```powershell
# 1. 가상환경 활성화
.\.venv\Scripts\Activate.ps1

# 2. 패키지 설치
pip install -r requirements.txt

# 3. Google Drive 데이터를 stft_image/에 배치

# 4. 데이터셋 생성
python dataset_prep.py

# 5. 학습 및 test 평가
python train_models/train_crnn.py
```

## Git에 포함하지 않는 파일

다음 항목은 용량이 크거나 실행 중 자동 생성되므로 저장소에 올리지 않습니다.

```text
stft_image/
dataset/
model_output/
.venv/
.idea/
*.npy
*.npz
*.pth
```

## 실험 결과 요약

기본 STFT와 미분 STFT를 12채널로 한 번에 입력하는 `combined` 방식과, 각각 6채널 분기로
학습한 뒤 결합하는 `dual` 방식을 비교했습니다. 테스트셋은 총 66개 샘플입니다.

| 모델 | Combined | Dual |
|---|---:|---:|
| CNN-RNN (CRNN) | 98.48% | 95.45% |
| CNN-TCN | 96.97% | 90.91% |
| EfficientNet | 96.97% | **100.00%** |
| MobileNetV3 | 60.61% | 63.64% |
| ResNet18 | **100.00%** | 92.42% |
| ShuffleNet | **100.00%** | 95.45% |

전체적으로 `combined` 방식이 더 안정적이었으며, Combined ResNet18·ShuffleNet과 Dual
EfficientNet이 가장 높은 정확도를 기록했습니다. MobileNetV3는 두 방식 모두 다른 모델보다
성능이 낮았습니다. 단, 테스트셋이 작고 클래스별 샘플 수가 불균형하므로 새로운 세션을 이용한
추가 검증이 필요합니다.

<초기 모델로 모델 개선해나갈 예정>
