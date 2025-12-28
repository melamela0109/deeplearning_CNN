
---

# CIFAR-10 Image Classification using ResNet-Lite

## 1. 프로젝트 개요

본 프로젝트는 딥러닝의 **ResNet(Residual Network)** 아키텍처를 경량화한 **ResNet-Lite** 모델을 직접 구현하여, CIFAR-10 이미지 데이터셋을 분류하는 모델을 구축했습니다.
단순한 CNN 구조에서 발생하는 기울기 소실(Vanishing Gradient) 문제를 잔차 연결(Residual Connection)로 해결하고, **파라미터 수와 데이터 수의 비율에 따른 모델 성능 변화**를 분석하는 데 중점을 두었습니다.

## 2. 개발 환경

* **OS:** Windows 10/11
* **Language:** Python 3.11.9
* **Framework:** PyTorch, Torchvision
* **Libraries:** NumPy, Matplotlib, PIL
* **Hardware:** NVIDIA GPU (CUDA 권장)

## 3. 파일 구조

```
├── main.py              # 전체 학습 및 평가 코드 (모델 구현, 학습 루프 포함)
├── cifar10_train.pth    # 학습 데이터 (별도 준비 필요)
├── cifar10_val.pth      # 검증 데이터 (별도 준비 필요)
├── cifar10_test.pth     # 테스트 데이터 (별도 준비 필요)
└── README.md            # 프로젝트 설명 파일

```

## 4. 모델 아키텍처 (ResNet-Lite)

CIFAR-10의 32x32 해상도에 최적화된 경량화된 ResNet 구조를 설계했습니다.

| Stage | Output Size | 상세 구성 |
| --- | --- | --- |
| **Stem** | 32x32 | Conv(3x3), BatchNorm, ReLU |
| **Layer 1** | 32x32 | Residual Block x 2 (Channel=32) |
| **Layer 2** | 16x16 | Residual Block x 2 (Channel=64) |
| **Layer 3** | 8x8 | Residual Block x 2 (Channel=128) |
| **Layer 4** | 4x4 | Residual Block x 2 (Channel=256) |
| **Head** | 1x1 | Global Avg Pool, Linear(10) |

## 5. 실행 방법

### 라이브러리 설치

```bash
pip install torch torchvision matplotlib numpy pillow

```

### 데이터 경로 설정

`main.py` 파일 내부의 `Config` 클래스에서 데이터가 위치한 경로를 수정해야 합니다.

```python
class Config:
    DATA_ROOT = r"C:\Your\Data\Path"  # 본인의 데이터 경로로 수정

```

### 학습 실행

```bash
python main.py

```

## 6. 실험 결과

* **Epochs:** 15
* **Batch Size:** 100
* **Optimizer:** Adam (Learning Rate: 0.001)

### 최종 성능

| 구분 | 정확도 (Accuracy) | 손실 (Loss) |
| --- | --- | --- |
| Train | 89.47% | 0.305 |
| Validation | 87.15% | 0.392 |
| **Test (Final)** | **87.17%** | - |

### 결과 분석

1. **ResNet 구조의 유효성:** 잔차 연결을 통해 깊은 망에서도 학습 효율이 저하되지 않고 15 Epoch 만에 87% 이상의 높은 정확도를 달성했습니다.
2. **Data Augmentation 효과:** Random Crop과 Horizontal Flip을 적용하여 Train 성능(89.47%)과 Test 성능(87.17%) 간의 격차를 줄이고 과적합을 방지했습니다.
3. **파라미터 수 vs 데이터 수:** 실험 결과, 파라미터 수가 데이터 수보다 많은 경우(Over-parameterized)에 적절한 규제가 동반될 때 가장 높은 성능을 보임을 확인했습니다.

---

