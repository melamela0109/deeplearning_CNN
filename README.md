# deeplearning_CNN
---

### 파일 구조 제안

```text
repo-root/
├── README.md          # 프로젝트 소개, 실행 방법, 요약
├── REPORT.md          # 상세 분석 보고서
├── main.py            # 소스 코드
├── requirements.txt   # 필요한 라이브러리 목록
└── assets/            # 결과 이미지들을 저장할 폴더
    ├── result_graph.png
    ├── case1_result.png
    └── ...

```

---

### 1. `README.md` (메인 소개 파일)

```markdown
# CIFAR-10 Classification using ResNet-Lite

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c) ![Status](https://img.shields.io/badge/Status-Finished-green)

## 프로젝트 개요
본 프로젝트는 ResNet(Residual Network) 아키텍처를 경량화한 `ResNet-Lite` 모델을 구축하여 CIFAR-10 데이터셋을 분류하는 딥러닝 프로젝트입니다. 
잔차 연결(Residual Connection)을 통해 깊은 망에서의 학습 효율을 높이고, **파라미터 수와 데이터 수의 비율에 따른 성능 변화**를 중점적으로 분석했습니다.

> **자세한 실험 결과와 분석 내용은 [상세 분석 보고서 (REPORT.md)](./REPORT.md)에서 확인하실 수 있습니다.**

## 모델 아키텍처 (ResNet-Lite)
기존 ResNet 구조를 CIFAR-10(32x32) 해상도에 맞춰 최적화하였습니다.

| Stage | Output Size | Configuration |
|:---:|:---:|:---|
| **Stem** | 32x32 | Conv(3x3), BN, ReLU |
| **Layer 1** | 32x32 | Residual Block x 2 (Ch=32) |
| **Layer 2** | 16x16 | Residual Block x 2 (Ch=64) |
| **Layer 3** | 8x8 | Residual Block x 2 (Ch=128) |
| **Layer 4** | 4x4 | Residual Block x 2 (Ch=256) |
| **Head** | 1x1 | Global Avg Pool, Linear(10) |

## 개발 환경 및 요구사항
* **OS:** Windows 10/11
* **Language:** Python 3.11.9
* **Library:** PyTorch, torchvision, NumPy, Matplotlib, PIL
* **Hardware:** NVIDIA GPU (CUDA Support recommended)

### 설치 방법
```bash
# 레포지토리 클론
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)

# 필수 라이브러리 설치
pip install torch torchvision matplotlib numpy pillow

```

## 실행 방법

1. 데이터 파일(`cifar10_train.pth`, `cifar10_val.pth`, `cifar10_test.pth`)을 준비합니다.
2. 소스 코드 내 `Config` 클래스의 `DATA_ROOT` 경로를 본인의 환경에 맞게 수정합니다.
3. 아래 명령어로 학습을 시작합니다.

```bash
python main.py

```

## 실험 결과 요약

* **Epochs:** 15
* **Batch Size:** 100
* **Optimizer:** Adam (LR=0.001)

| Metric | Score |
| --- | --- |
| **Train Accuracy** | 89.47% |
| **Validation Accuracy** | 87.15% |
| **Final Test Accuracy** | **87.17%** |

### 주요 성과

* **Over-parameterization 효과 입증:** 파라미터 수가 데이터 수보다 월등히 많을 때(Case 1) 가장 높은 성능을 보임.
* **ResNet 구조의 유효성:** 15 Epoch 만에 87% 이상의 정확도 달성 및 안정적인 수렴 확인.
* **과적합 방지:** Data Augmentation(Crop, Flip)을 통해 Train/Test 간 격차를 최소화.

---

© 2025. Project by [Your Name].

```

---

### 2. `REPORT.md` (상세 분석 보고서)

```markdown
# 상세 분석 보고서: CIFAR-10 Classification

## I. 프로젝트 개요

### 1. 목표
본 프로젝트의 핵심 목표는 딥러닝의 CNN 구조를 기반으로 CIFAR-10 이미지 데이터셋을 높은 정확도로 분류하는 고성능 예측 모델을 구축하는 것입니다. 기울기 소실 문제를 해결하기 위해 ResNet(Residual Network) 아키텍처를 채택하였으며, 실험 환경에 맞춰 파라미터 효율성을 고려한 'ResNet-Lite' 모델을 설계했습니다.

### 2. 데이터셋 (CIFAR-10)
* **구성:** 10개 클래스 (Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck)
* **규격:** 32×32 Pixel, 3 Channel (RGB)
* **분할:** Train(40,000), Validation(10,000), Test(10,000)
* **로드 방식:** `CustomCIFAR10` 클래스를 구현하여 `.pth` 파일을 직접 로드 및 처리

---

## II. 실험 환경 및 설정

### 1. 개발 환경
* **OS:** Windows 10/11 (64-bit)
* **Framework:** PyTorch (CUDA 11.8 지원)
* **Library:** NumPy, Matplotlib, Pillow

### 2. 하이퍼파라미터
| 항목 | 설정값 | 비고 |
|---|---|---|
| Batch Size | 100 | GPU 효율 및 학습 속도 고려 |
| Learning Rate | 0.001 | Adam Optimizer 권장값 |
| Epochs | 15 | 과적합 및 수렴 속도 고려 |
| Seed | 42 | 결과 재현성 확보 |

---

## III. 모델 설계 및 구현

### 1. 데이터 전처리
* **정규화(Normalization):** 입력 이미지를 0~1로 변환 후 평균 0.5, 표준편차 0.5 적용.
* **데이터 증강(Data Augmentation):**
    * Random Crop (padding=4)
    * Random Horizontal Flip (p=0.5)

### 2. ResNet-Lite 아키텍처
기존 CNN의 기울기 소실 문제를 해결하기 위해 잔차 블록(Residual Block)을 도입했습니다.

* **Hierarchical Feature Extraction:** 4개의 Stage를 거치며 채널 수 확장 (32 → 64 → 128 → 256)
* **Skip Connection:** $F(x) + x$ 구조를 통해 정보 손실 최소화
* **GAP (Global Average Pooling):** 파라미터 수가 많은 FC Layer 대신 사용하여 연산 효율 증대 및 과적합 방지

---

## IV. 실험 결과

### 1. 학습 진행 결과 (Log)
| Epoch | Train Acc | Val Acc | Val Loss | 상태 |
|:---:|:---:|:---:|:---:|:---|
| 1 | 46.56% | 49.29% | 1.391 | 초기 학습 |
| 5 | 78.45% | 77.14% | 0.682 | 70%대 진입 |
| 10 | 85.87% | 83.41% | 0.478 | 안정화 단계 |
| **15** | **89.47%** | **87.15%** | **0.392** | **최종 수렴** |

### 2. 최종 테스트 결과
* **Test Accuracy:** **87.17%**
* **분석:** ResNet 구조의 우수성과 데이터 증강 효과로 인해, 일반적인 CNN(70% 중반) 대비 약 10%p 향상된 성능을 달성하였습니다.

---

## V. 결과 분석 및 고찰

### 1. 모델 크기(Parameter) vs 데이터 수 비교
파라미터 수(P)와 데이터 수(D)의 비율에 따른 성능 변화를 실험하였습니다.

| 실험군 | 파라미터 수 | 데이터 수 | 상태 | Test Acc |
|:---:|:---:|:---:|:---:|:---:|
| **Case 1** | **2,797,610** | **40,000** | **Over (P≫D)** | **87.17% (Best)** |
| Case 2 | 44,370 | 40,000 | Balanced (P≈D) | 74.89% |
| Case 3 | 25,810 | 40,000 | Under (P<D) | 70.75% |

> **Insight:** 통계적 통념과 달리, 딥러닝에서는 **Over-parameterized(과매개변수화)** 된 모델이 적절한 규제(Augmentation 등)와 결합될 때 가장 뛰어난 성능과 일반화 능력을 보였습니다.

### 2. 성능 개선 방안
* **활성화 함수:** ReLU 대신 GELU(Gaussian Error Linear Unit) 도입 고려.
* **Scheduler:** Cosine Annealing 등을 통해 학습 후반부 학습률 미세 조정.

---

## VI. 참고문헌
1. baek2sm. (2021). 딥러닝 CNN 모델 살펴보기(4): ResNet 논문 리뷰.
2. IBM. (n.d.). 데이터 증강이란 무엇인가요?. IBM Think.
3. hyuno. (2022). [논문리뷰] Deep Double Descent. Velog.
4. daebaq27. (2021). 최적화, 경사하강법 (ResNet). Tistory.

```
