import os
import time
import multiprocessing
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# =============================================================================
# 1. 설정 및 하이퍼파라미터 (Configuration)
# =============================================================================
class Config:
    """실험에 사용되는 모든 경로와 하이퍼파라미터를 관리하는 클래스"""
    PROJECT_NAME = "CIFAR10(파라미터 수 > 데이터 수)"
    DATA_ROOT = r"C:\deep learnnng"  # 데이터 경로
    
    # 하이퍼파라미터
    BATCH_SIZE = 100
    LEARNING_RATE = 0.001
    EPOCHS = 15
    SEED = 42
    
    # 시스템 설정
    NUM_WORKERS_TRAIN = 2
    NUM_WORKERS_TEST = 0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 시드 고정 (재현성 확보)
torch.manual_seed(Config.SEED)

# =============================================================================
# 2. 데이터셋 클래스 (Custom Dataset)
# =============================================================================
class CustomCIFAR10(Dataset):
    def __init__(self, path: str, transform=None):
        # weights_only=False: numpy 배열 등 데이터 구조 로딩 허용
        data_dict = torch.load(path, map_location='cpu', weights_only=False)
        
        self.data = data_dict['data']       # (N, 32, 32, 3) uint8
        self.targets = data_dict['targets'] # (N,) int64
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, target

# =============================================================================
# 3. 모델 아키텍처 (Model Architecture)
# =============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNetLite(nn.Module):
    def __init__(self):
        super(ResNetLite, self).__init__()
        # Stem
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Stages
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.layer4 = self._make_layer(128, 256, 2, stride=2)
        
        # Head
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)

    def _make_layer(self, in_c, out_c, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_c, out_c, stride))
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_c, out_c, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# =============================================================================
# 4. 유틸리티 함수 (Training & Analysis)
# =============================================================================
def count_parameters(model):
    """모델의 총 파라미터 개수를 계산하는 함수"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device).long()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / len(loader), 100 * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / len(loader), 100 * correct / total

def save_plots(train_losses, val_losses, train_accs, val_accs, filename="cifar10_result.png"):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='royalblue')
    plt.plot(val_losses, label='Val Loss', color='darkorange')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Loss Trend'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc', color='royalblue')
    plt.plot(val_accs, label='Val Acc', color='darkorange')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)'); plt.title('Accuracy Trend'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"\n[결과 그래프 저장 완료] {os.path.abspath(filename)}")

# =============================================================================
# 5. 메인 실행 로직 (Main)
# =============================================================================
def main():
    print(f"Project: {Config.PROJECT_NAME}")
    print(f"Device : {Config.DEVICE}")

    # 1. 데이터 전처리
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 2. 데이터 로드
    try:
        train_ds = CustomCIFAR10(os.path.join(Config.DATA_ROOT, 'cifar10_train.pth'), transform=train_transform)
        val_ds   = CustomCIFAR10(os.path.join(Config.DATA_ROOT, 'cifar10_val.pth'), transform=test_transform)
        test_ds  = CustomCIFAR10(os.path.join(Config.DATA_ROOT, 'cifar10_test.pth'), transform=test_transform)

        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,  num_workers=Config.NUM_WORKERS_TRAIN)
        val_loader   = DataLoader(val_ds,   batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS_TEST)
        test_loader  = DataLoader(test_ds,  batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS_TEST)
        print(f"Data Loaded: Train({len(train_ds)}), Val({len(val_ds)}), Test({len(test_ds)})")

    except Exception as e:
        print(f"\n[Error] 데이터 파일 경로 확인 필요: {e}")
        return

    # 3. 모델 초기화
    model = ResNetLite().to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # =========================================================================
    # 파라미터 수 vs 데이터 수 비교 분석
    # =========================================================================
    num_params = count_parameters(model)
    num_data = len(train_ds)

    print("-" * 50)
    print(f"[모델 분석]")
    print(f" - 총 파라미터 수 : {num_params:,} 개")
    print(f" - 학습 데이터 수 : {num_data:,} 개")
    
    if num_params > num_data:
        print(f" [상태] Over-parameterized (파라미터 > 데이터)")
        print(f" (최신 딥러닝에서는 일반적이나, 과적합 방지를 위해 Augmentation이 필수적임)")
    else:
        print(f" [상태] Under-parameterized (데이터 > 파라미터)")
    print("-" * 50)

    # 4. 학습 루프
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0
    start_time = time.time() # 시간 측정 시작

    print("\n학습 시작...")
    for epoch in range(Config.EPOCHS):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        v_loss, v_acc = evaluate(model, val_loader, criterion, Config.DEVICE)
        
        history['train_loss'].append(t_loss); history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc); history['val_acc'].append(v_acc)
        
        # 최고 성능 모델 저장 (Start Save 메시지 출력 제거)
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), "best_model.pth") 

        print(f"Epoch [{epoch+1:02d}/{Config.EPOCHS}] "
              f"Train: {t_acc:.2f}% | Val: {v_acc:.2f}%")

    total_time = time.time() - start_time
    print(f"\n총 학습 시간: {total_time:.2f}초 (약 {total_time/60:.1f}분)")

    # 5. 결과 시각화
    save_plots(history['train_loss'], history['val_loss'], 
               history['train_acc'], history['val_acc'])

    # 6. 최종 평가 (Best Model 로드 후 평가)
    print("\n최종 테스트 평가 (Best Model 사용)...")
    # 저장된 최고의 가중치 불러오기
    model.load_state_dict(torch.load("best_model.pth", weights_only=True)) 
    test_loss, test_acc = evaluate(model, test_loader, criterion, Config.DEVICE)
    
    print("=" * 50)
    print(f"최종 Test Accuracy: {test_acc:.2f}%")
    print("=" * 50)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()