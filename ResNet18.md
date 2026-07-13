6축 로봇팔 STFT 기반 ResNet18 이상 탐지 <베타 테스트>
1. 프로젝트 목표

6축 로봇팔에서 측정한 3상 전류 데이터를 STFT로 변환하고, ResNet18 기반 CNN을 이용해 각 축의 이상 여부를 탐지하는 것을 목표로 한다.

모델은 하나의 샘플에서 6개 축의 데이터를 동시에 입력받고, 각 축에 대한 이상 확률을 출력한다.

입력  : 6축 × 3상 전류 STFT
출력  : 축 1~6의 이상 확률

예를 들어 모델 출력이 다음과 같다면:

[0.02, 0.05, 0.91, 0.04, 0.08, 0.03]

3번 축의 이상 가능성이 가장 높다고 판단한다.

2. STFT 데이터 구조

각 로봇 축의 전류 데이터는 R상, S상, T상의 3상 전류로 구성된다.

한 축의 STFT 데이터 형태는 다음과 같다.

(3, 1025, 64)

각 차원의 의미는 다음과 같다.

3    : R상, S상, T상
1025 : 주파수 bin
64   : 시간 bin

6개 축의 STFT 데이터를 동일한 실험과 동일한 시간 구간을 기준으로 결합하면 하나의 샘플은 다음 형태가 된다.

(6, 3, 1025, 64)

각 차원의 의미는 다음과 같다.

6    : 로봇팔의 6개 관절 축
3    : 각 축의 3상 전류
1025 : 주파수 bin
64   : 시간 bin

6축과 3상을 하나의 18채널로 합치지 않고, 축과 상의 차원을 분리한 상태로 유지하였다.

3. STFT 데이터 불러오기

STFT로 변환된 .npy 파일은 다음 폴더에서 불러왔다.

normal_image/
abnormal_image/
normal_image: 정상 상태의 STFT 데이터
abnormal_image: 이상 상태의 STFT 데이터

각 데이터의 형태가 (F, T, 3)인 경우 다음 형태로 변환하였다.

(3, F, T)

이를 통해 모든 축 데이터의 차원 순서를 다음과 같이 통일하였다.

(상, 주파수, 시간)
4. 6축 데이터 적층

파일 이름에서 실험 ID, chunk 번호, 축 번호를 추출하고, 동일한 실험과 동일한 chunk에 해당하는 6개 축 데이터를 매칭하였다.

6개 축이 모두 존재하는 경우에만 다음 방식으로 적층하였다.

np.stack(axis_data, axis=0)

최종 적층 결과는 다음과 같다.

Normal samples: 318
Normal first shape: (6, 3, 1025, 64)

Abnormal samples: 310
Abnormal first shape: (6, 3, 1025, 64)

따라서 정상 샘플 318개와 이상 샘플 310개가 생성되었다.

5. 라벨 구성

각 샘플은 6개 축의 이상 여부를 나타내는 다중 라벨을 사용한다.

정상 데이터의 라벨은 다음과 같다.

[0, 0, 0, 0, 0, 0]

3번 축에 이상이 있는 경우:

[0, 0, 1, 0, 0, 0]

2번과 5번 축에 이상이 있는 경우:

[0, 1, 0, 0, 1, 0]

각 축은 서로 독립적으로 이상 상태가 될 수 있으므로, 이 문제는 다중 클래스 분류가 아니라 다중 라벨 분류로 구성한다.

6. 데이터 저장

적층된 데이터와 라벨은 .npz 파일로 저장하였다.

각 파일에는 다음 정보가 포함된다.

data       : (6, 3, 1025, 64)
label      : (6,)
sample_key : 실험과 chunk를 구분하는 식별자

저장 폴더는 다음과 같이 구성하였다.

dataset/
├── normal/
└── abnormal/

정상 데이터는 dataset/normal, 이상 데이터는 dataset/abnormal에 저장하였다.

7. 학습·검증·테스트 분할

저장된 데이터를 다음 비율로 분할하였다.

Train      : 70%
Validation : 15%
Test       : 15%


최종 학습 전에는 모든 이상 데이터가 train, validation, test 중 하나에 포함되는지 검사해야 한다.

또한 동일한 원본 실험에서 생성된 여러 chunk가 서로 다른 데이터 분할에 들어가지 않도록, 최종적으로는 실험 단위 분할을 적용해야 한다.

8. PyTorch Dataset과 DataLoader

저장된 .npz 파일을 PyTorch에서 사용하기 위해 Dataset 클래스를 구성하였다.

Dataset은 각 파일에서 다음 정보를 불러온다.

data
label

원본 STFT의 주파수 크기는 1025이지만, ResNet18의 연산 과정에서 크기를 쉽게 맞추기 위해 마지막 주파수 bin 하나를 제거한다.

기존 형태 : (6, 3, 1025, 64)
변환 형태 : (6, 3, 1024, 64)

배치 차원을 포함한 DataLoader 출력 형태는 다음과 같다.

(B, 6, 3, 1024, 64)

여기서 B는 batch size이다.

Train DataLoader는 shuffle=True
Validation DataLoader는 shuffle=False
Test DataLoader는 shuffle=False

로 설정한다.

9. 공유 ResNet18 구조

6개 축에 각각 별도의 ResNet18을 만드는 대신, 하나의 ResNet18을 모든 축에 공통으로 적용한다.

축 1 STFT → 공유 ResNet18 → 특징 벡터 z1
축 2 STFT → 공유 ResNet18 → 특징 벡터 z2
축 3 STFT → 공유 ResNet18 → 특징 벡터 z3
축 4 STFT → 공유 ResNet18 → 특징 벡터 z4
축 5 STFT → 공유 ResNet18 → 특징 벡터 z5
축 6 STFT → 공유 ResNet18 → 특징 벡터 z6

ResNet18의 마지막 분류층을 제거하면 각 축에서 512차원 특징 벡터가 생성된다.

입력 형태:

(B, 6, 3, 1024, 64)

축별 특징 형태:

(B, 6, 512)

각 축 특징에 동일한 분류층을 적용하면 최종 출력은 다음과 같다.

(B, 6)

각 출력값은 해당 축의 이상 여부를 판단하기 위한 logit이다.

10. 손실 함수와 옵티마이저

6개 축을 독립적으로 정상 또는 이상으로 판단하므로 손실 함수는 다음을 사용한다.

BCEWithLogitsLoss

BCEWithLogitsLoss는 Sigmoid와 Binary Cross Entropy를 결합한 손실 함수이다.

모델 출력에는 별도의 Sigmoid를 적용하지 않고 logits 상태로 손실 함수에 전달한다.

옵티마이저는 다음을 사용한다.

Adam
11. 학습 및 검증 과정

학습 과정에서는 다음 순서로 처리한다.

1. 배치 데이터를 device로 이동
2. 모델에 입력
3. logits 계산
4. BCEWithLogitsLoss 계산
5. 기울기 초기화
6. 역전파
7. 가중치 업데이트

검증 과정에서는 가중치를 변경하지 않으며 다음을 사용한다.

torch.no_grad()

각 epoch마다 다음 값을 출력한다.

Train Loss
Validation Loss

Validation Loss가 가장 낮은 모델의 가중치를 저장한다.

12. 전체 처리 흐름

현재까지 구성된 전체 데이터 처리 및 모델 흐름은 다음과 같다.

3상 전류 데이터
        ↓
STFT 변환
        ↓
축별 STFT 생성
(3, 1025, 64)
        ↓
동일 시간대의 6축 적층
(6, 3, 1025, 64)
        ↓
축별 이상 라벨 연결
(6,)
        ↓
NPZ 파일 저장
        ↓
Train / Validation / Test 분할
        ↓
PyTorch Dataset 및 DataLoader
        ↓
주파수 bin 하나 제거
(B, 6, 3, 1024, 64)
        ↓
공유 ResNet18 특징 추출
(B, 6, 512)
        ↓
축별 분류층
(B, 6)
        ↓
BCEWithLogitsLoss
        ↓
6개 축의 이상 여부 판단
