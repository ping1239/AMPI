# 모터 전류 데이터 기반 이상 탐지 및 분류 파이프라인

이 프로젝트는 모터에서 측정된 3상 전류 데이터(.mat)를 STFT(Short-Time Fourier Transform)를 통해 주파수 스펙트로그램 이미지 텐서로 변환하고, 이를 사용하여 정상/비정상 상태를 탐지 및 분류하는 종합 딥러닝 파이프라인입니다.

비지도학습 기반의 **오토인코더(Autoencoder)** 탐지 모델과, 지도학습 기반의 다양한 **CNN/RNN/TCN 및 경량화 분류 모델**들을 모두 제공합니다.

---

## 📂 주요 폴더 구조 및 역할

*   **`normal_data/`** : 정상 상태에서 측정된 원본 모터 3상 전류 데이터 (`.mat` 파일)가 보관되는 폴더입니다.
*   **`abnormal_data/`** : 고장(비정상) 상태에서 측정된 원본 모터 3상 전류 데이터 (`.mat` 파일)가 보관되는 폴더입니다.
*   **`normal_image/`** : `STFT.ipynb` 코드를 거쳐 정상 데이터가 딥러닝용 텐서(`.npy`)로 변환되어 저장되는 폴더입니다.
*   **`abnormal_image/`** : `STFT.ipynb` 코드를 거쳐 비정상 데이터가 딥러닝용 텐서(`.npy`)로 변환되어 저장되는 폴더입니다.
*   **`sql_connect/`** : 학습 결과나 이력을 MySQL 데이터베이스에 기록하기 위한 데이터베이스 연결 코드 폴더입니다.
*   **모델 폴더들 (지도학습 분류)** :
    *   **`ResNet18/`** : PyTorch 기반의 ResNet-18 분류 파이프라인
    *   **`CNN-RNN/`** : 시계열 흐름을 학습하는 CRNN (CNN + GRU) 분류 파이프라인
    *   **`CNN-TCN/`** : Temporal Convolutional Network 기반의 분류 파이프라인
    *   **`EfficientNet/`** : 고효율 특징 추출 백본인 EfficientNet 분류 파이프라인
    *   **`MobileNetV3/`** : 경량 실시간 탐지용 MobileNetV3 분류 파이프라인
    *   **`ShuffleNet/`** : 연산량을 극대화해 줄인 ShuffleNet 분류 파이프라인

### 🌳 디렉토리 구조 (Directory Tree)
필수 파일들로 구성된 프로젝트의 전체적인 폴더 구조는 아래와 같습니다.

```text
Project_Root/
├── STFT.ipynb            # 데이터 전처리 및 정규화(STFT 변환) 스크립트
├── Auto_encoder.ipynb    # 오토인코더 모델 학습 및 이상 탐지 스크립트
├── ResNet18.ipynb        # ResNet18 테스트 및 검증 노트북
├── requirements.txt      # 프로젝트 실행을 위한 의존성 패키지 목록
│
├── CNN-RNN/              # CRNN 모델 파트
│   ├── dataset_prep.py   # CRNN 학습을 위한 슬라이딩 윈도우 데이터 전처리
│   └── CNN-RNN.py        # 모델 학습, 검증, 그래프 시각화 및 평가
│
├── CNN-TCN/              # CNN-TCN 모델 파트
│   ├── dataset_prep.py   # CNN-TCN 학습을 위한 데이터 전처리
│   └── CNN-TCN.py        # 모델 학습, 검증, 그래프 시각화 및 평가
│
├── EfficientNet/         # EfficientNet 모델 파트
│   ├── dataset_prep.py
│   └── train_efficientnet.py
│
├── MobileNetV3/          # MobileNetV3 모델 파트
│   ├── dataset_prep.py
│   └── train_mobilenetv3.py
│
├── ShuffleNet/           # ShuffleNet 모델 파트
│   ├── dataset_prep.py
│   └── train_shufflenet.py
│
├── ResNet18/             # ResNet18 모델 파트
│   ├── dataset_prep.py
│   └── train_resnet.py
│
└── sql_connect/          # 데이터베이스 연동 파트
    └── mysql_connect.py  # MySQL 연결 및 학습 로그 저장 스크립트
```

---

## 🚀 실행 흐름 (Workflow)

프로젝트는 크게 **[데이터 전처리 및 변환]** 단계와 **[모델 선정 및 학습/평가]** 단계로 구분됩니다.

### 단계 1: 데이터 공통 전처리 (`STFT.ipynb`)
모든 딥러닝 모델의 학습에 쓰일 원시 데이터를 이미지 텐서로 변환하는 단계입니다.
1. Jupyter 환경에서 **`STFT.ipynb`** 파일을 엽니다.
2. 위에서부터 순서대로 셀을 실행하여 `normal_data/`와 `abnormal_data/` 내부의 `.mat` 파일들을 읽어들입니다.
3. 신호를 STFT 변환을 거쳐 주파수 스펙트로그램 이미지로 변환 및 절대 정규화를 거쳐 텐서 파일(`normal_image.npy`, `abnormal_image.npy`)로 저장합니다.

---

### 단계 2-A: 비지도학습 이상 탐지 (`Auto_encoder.ipynb`)
정상 데이터의 특징만을 압축 및 복원하는 훈련을 통해 고장 데이터를 솎아내는 과정입니다.
1. **`Auto_encoder.ipynb`** 파일을 열고 순서대로 셀을 실행합니다.
2. **모델 학습:** 정상 데이터(`normal_image.npy`)만 오토인코더에 입력하여 복원 특징을 훈련합니다.
3. **임계값 설정:** 정상 복원 시의 오차 기준선(Threshold)을 수립합니다.
4. **이상 탐지:** 검증용 고장 데이터를 넣었을 때 임계값을 초과하는 복원 오차가 발생하는 것을 확인하여 비정상 상태를 최종 판별합니다.

---

### 단계 2-B: 지도학습 정상/이상 분류 (CNN, CRNN, TCN 등)
각 모델 폴더별로 독립된 지도학습 데이터 준비 및 학습 프로세스가 진행됩니다. (예: `CNN-TCN` 학습 시)

1. **데이터셋 격리 준비**:
   각 모델 폴더 하위의 `dataset_prep.py` 스크립트를 실행하여 데이터 누수가 없는 Train(70%)/Val(15%)/Test(15%) 데이터셋 조각을 만듭니다.
   ```bash
   python CNN-TCN/dataset_prep.py
   ```
2. **모델 학습 및 검증**:
   모델 학습 스크립트를 실행하여 학습을 시작합니다.
   ```bash
   python CNN-TCN/CNN-TCN.py
   ```
3. 학습이 진행되며 최고 검증 정확도 도달 시 가중치 파일(`.pth`)이 `model_output/`에 자동 보관되고, 에포크별 Loss/Accuracy 추이 그래프(학습 곡선)가 저장 및 팝업됩니다.
4. 마지막으로 Test Set으로 학습에 참여하지 않은 데이터를 사용하여 최종 분류 성능 정확도를 계산합니다.

---

## ⚙️ 환경 세팅 가이드

이 프로젝트를 실행하기 위해 필요한 Python 라이브러리 패키지는 `requirements.txt`에 정리되어 있습니다.
터미널을 열고 아래 명령어를 실행하여 필요한 패키지를 일괄 설치해 주세요.

```bash
# 가상환경 활성화 (프로젝트 폴더 내에 .venv가 있는 경우)
source .venv/bin/activate

# 패키지 일괄 설치
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 💾 데이터베이스 연동 가이드 (선택사항)

학습 완료 후 성능 기록을 시스템에 자동으로 저장하고 싶다면 MySQL 데이터베이스를 연동할 수 있습니다.

1. `sql_connect/mysql_connect.py` 파일 상단의 `DB_CONFIG` 정보를 본인의 MySQL 서버 사양에 맞게 수정합니다.
2. 테이블을 자동 생성하거나 작동 여부를 확인하기 위해 테스트 모듈을 실행합니다.
   ```bash
   python sql_connect/mysql_connect.py
   ```