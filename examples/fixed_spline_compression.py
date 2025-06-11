import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from tqdm import tqdm

def get_model_size_mb(model):
    """모델 크기를 MB 단위로 계산"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

class CorrectSplineLinear(nn.Module):
    """올바른 스플라인 압축 - 각 출력별로 입력에 대한 제어점만 저장"""
    
    def __init__(self, in_features: int, out_features: int, 
                 k: int = 3,  # 제어점 개수 - 1
                 initial_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        
        # 올바른 방식: 각 출력마다 k+1개의 제어점만 저장
        # control_points: (out_features, k+1) - 입력 인덱스에 대한 제어점
        self.control_points = nn.Parameter(torch.randn(out_features, k + 1))
        
        # 제어점 값들: (out_features, k+1, compressed_dim)
        # compressed_dim을 입력 차원보다 작게 설정하여 실제 압축 달성
        self.compressed_dim = max(1, in_features // 4)  # 4배 압축
        self.control_values = nn.Parameter(torch.randn(out_features, k + 1, self.compressed_dim))
        
        # 압축된 차원을 원래 차원으로 복원하는 매핑
        self.expansion_matrix = nn.Parameter(torch.randn(self.compressed_dim, in_features))
        
        if initial_weight is not None:
            self._fit_to_weight(initial_weight)
    
    def _fit_to_weight(self, target_weight):
        """기존 가중치에 맞춰 제어점 최적화"""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        
        for step in range(100):  # 빠른 피팅
            optimizer.zero_grad()
            reconstructed = self._reconstruct_weight()
            loss = F.mse_loss(reconstructed, target_weight)
            loss.backward()
            optimizer.step()
            
            if step % 20 == 0:
                print(f"    피팅 Step {step}, Loss: {loss.item():.6f}")
    
    def _reconstruct_weight(self):
        """제어점으로부터 가중치 복원"""
        # 각 출력에 대해 스플라인 보간으로 가중치 생성
        weights = []
        
        for i in range(self.out_features):
            # i번째 출력의 제어점들
            control_pts = self.control_points[i]  # (k+1,)
            control_vals = self.control_values[i]  # (k+1, compressed_dim)
            
            # 입력 차원에 대한 균등 분할 점들
            t_points = torch.linspace(0, 1, self.in_features, device=control_pts.device)
            
            # 각 입력 위치에서 보간된 값 계산
            interpolated_compressed = []
            for t in t_points:
                # 스플라인 보간으로 압축된 값 계산
                interp_val = self._spline_interpolate(control_pts, control_vals, t)
                interpolated_compressed.append(interp_val)
            
            # (in_features, compressed_dim)
            interpolated_compressed = torch.stack(interpolated_compressed, dim=0)
            
            # 압축된 차원을 원래 차원으로 복원
            weight_row = torch.sum(interpolated_compressed.unsqueeze(-1) * self.expansion_matrix.unsqueeze(0), dim=1)
            weights.append(weight_row)
        
        return torch.stack(weights, dim=0)  # (out_features, in_features)
    
    def _spline_interpolate(self, control_points, control_values, t):
        """단순 선형 보간 (스플라인 대신)"""
        k = len(control_points) - 1
        
        # t를 [0, k] 범위로 스케일링
        t_scaled = t * k
        
        # 이웃한 두 제어점 찾기
        j = torch.clamp(torch.floor(t_scaled), 0, k-1).long()
        t_local = t_scaled - j
        
        # 선형 보간
        if j >= k:
            return control_values[-1]
        else:
            return control_values[j] * (1 - t_local) + control_values[j + 1] * t_local
    
    def forward(self, x):
        """Forward pass"""
        weight = self._reconstruct_weight()
        return F.linear(x, weight)

def convert_to_correct_spline(model, k=3):
    """모델을 올바른 스플라인 압축으로 변환"""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            print(f"  🔧 {name}: {module.out_features}×{module.in_features}")
            
            # 원본 파라미터 수
            original_params = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
            
            # 새로운 스플라인 레이어
            spline_layer = CorrectSplineLinear(
                module.in_features, 
                module.out_features, 
                k=k,
                initial_weight=module.weight.data
            )
            
            # 압축된 파라미터 수
            compressed_params = sum(p.numel() for p in spline_layer.parameters())
            compression_ratio = compressed_params / original_params
            
            print(f"    원본: {original_params:,} → 압축: {compressed_params:,}")
            print(f"    압축률: {compression_ratio:.3f} ({compression_ratio*100:.1f}%)")
            
            # 바이어스 처리
            if module.bias is not None:
                spline_layer.bias = nn.Parameter(module.bias.data.clone())
            
            setattr(model, name, spline_layer)
        
        else:
            # 재귀적 처리
            convert_to_correct_spline(module, k)

def demo_correct_compression():
    """올바른 스플라인 압축 데모"""
    print("🚀 올바른 스플라인 압축 데모")
    
    # 중간 크기 모델로 테스트
    class TestMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(512, 1024)  # 0.5M 파라미터
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(1024, 512)  # 0.5M 파라미터
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(512, 128)   # 65K 파라미터
            self.fc4 = nn.Linear(128, 10)    # 1.3K 파라미터

        def forward(self, x):
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = self.relu2(self.fc3(x))
            return self.fc4(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")
    
    # 원본 모델
    original_model = TestMLP().to(device)
    test_input = torch.randn(16, 512, device=device)
    
    with torch.no_grad():
        original_output = original_model(test_input)
    
    original_params = sum(p.numel() for p in original_model.parameters())
    original_size_mb = get_model_size_mb(original_model)
    
    print(f"\n📊 원본 모델:")
    print(f"  파라미터: {original_params:,}")
    print(f"  크기: {original_size_mb:.2f} MB")
    
    # 스플라인 압축 적용
    print(f"\n🔧 올바른 스플라인 압축 적용:")
    compressed_model = TestMLP().to(device)
    compressed_model.load_state_dict(original_model.state_dict())
    
    convert_to_correct_spline(compressed_model, k=3)
    
    # 결과 비교
    with torch.no_grad():
        compressed_output = compressed_model(test_input)
    
    compressed_params = sum(p.numel() for p in compressed_model.parameters())
    compressed_size_mb = get_model_size_mb(compressed_model)
    
    compression_ratio = compressed_params / original_params
    accuracy = F.cosine_similarity(original_output.flatten(), compressed_output.flatten(), dim=0).item()
    mse = F.mse_loss(original_output, compressed_output).item()
    
    print(f"\n📈 최종 결과:")
    print(f"  압축 파라미터: {compressed_params:,}")
    print(f"  압축 크기: {compressed_size_mb:.2f} MB")
    print(f"  압축률: {compression_ratio:.3f} ({compression_ratio*100:.1f}%)")
    print(f"  크기 감소: {(1-compression_ratio)*100:.1f}%")
    print(f"  정확도: {accuracy:.4f}")
    print(f"  MSE: {mse:.6f}")
    
    if compression_ratio < 1.0:
        print(f"  🎉 압축 성공!")
    else:
        print(f"  ❌ 압축 실패")

if __name__ == "__main__":
    demo_correct_compression() 