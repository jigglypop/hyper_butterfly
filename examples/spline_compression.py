import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers.modeling_utils import Conv1D
from datasets import load_dataset
HF_AVAILABLE = True
RS_AVAILABLE = True

try:
    import reality_stone as rs
    print("✅ reality_stone 라이브러리 사용 가능")
except ImportError:
    RS_AVAILABLE = False
    raise Exception("reality_stone 라이브러리를 찾을 수 없음")

def catmull_rom_interpolation(control_points, t_values):
    """벡터화된 Catmull-Rom 스플라인 보간"""
    k = control_points.shape[0] - 1
    m = t_values.shape[0]
    
    t_scaled = t_values * k
    j = torch.floor(t_scaled).long()
    t_local = t_scaled - j.float()
    
    # 인덱스가 범위를 벗어나지 않도록 클램핑
    j = torch.clamp(j, 1, k - 2)
    
    p0 = control_points[j - 1]
    p1 = control_points[j]
    p2 = control_points[j + 1]
    p3 = control_points[j + 2]
    
    t_local = t_local.unsqueeze(1) # (m, 1)
    
    # Catmull-Rom 공식
    return 0.5 * (
        (2 * p1) +
        (-p0 + p2) * t_local +
        (2 * p0 - 5 * p1 + 4 * p2 - p3) * t_local.pow(2) +
        (-p0 + 3 * p1 - 3 * p2 + p3) * t_local.pow(3)
    )

def geodesic_spline_with_reality_stone(control_points, t_values, use_reality_stone=True):
    """벡터화된 지오데식 스플라인 보간 (reality_stone 사용)"""
    if not RS_AVAILABLE or not use_reality_stone:
        return catmull_rom_interpolation(control_points, t_values)
    
    try:
        k = control_points.shape[0] - 1
        m = t_values.shape[0]
        
        t_scaled = t_values * k
        j = torch.floor(t_scaled).long()
        t_local = (t_scaled - j.float()).unsqueeze(-1) # (m, 1)

        j = torch.clamp(j, 1, k - 2)
        
        # 제어점 선택 (m, in_features)
        p0 = control_points[j - 1]
        p1 = control_points[j]
        p2 = control_points[j + 1]
        p3 = control_points[j + 2]

        # 참고: 실제 reality_stone의 함수명은 라이브러리 버전에 따라 다를 수 있습니다.
        # 예를 들어 lorentz_exp_map, poincare_exp_map 등이 될 수 있습니다.
        # 여기서는 lorentz_exp_map/log_map을 사용한다고 가정합니다.
        exp_map_func = getattr(rs, 'lorentz_exp_map', None)
        log_map_func = getattr(rs, 'lorentz_log_map', None)

        if exp_map_func and log_map_func:
            # 접선 벡터 계산 (m, in_features)
            v1 = log_map_func(p1, p2)
            v0 = log_map_func(p1, p0)
            v2 = log_map_func(p1, p3)
            
            # Hermite 기반 접선 벡터
            tangent_p1 = 0.5 * (v1 - v0)
            tangent_p2 = 0.5 * (v2 - v1)
            
            # Hermite 계수 (m, 1)
            h00 = 2 * t_local**3 - 3 * t_local**2 + 1
            h10 = t_local**3 - 2 * t_local**2 + t_local
            h01 = -2 * t_local**3 + 3 * t_local**2
            h11 = t_local**3 - t_local**2
            
            # 접선 공간에서 보간
            tangent_interp = h10 * tangent_p1 + h01 * v1 + h11 * tangent_p2
            
            # 지수 맵으로 다시 매니폴드로
            result = exp_map_func(p1, tangent_interp)
            return result
        else:
            print("reality_stone에서 'lorentz_exp_map'/'lorentz_log_map'을 찾을 수 없습니다. Fallback합니다.")
            return catmull_rom_interpolation(control_points, t_values)
            
    except Exception as e:
        import traceback
        print(f"Reality Stone 지오데식 보간 실패: {e}. 일반 스플라인 사용.")
        traceback.print_exc()
        return catmull_rom_interpolation(control_points, t_values)

class SplineLinearTransform(nn.Module):
    """
    스플라인 기반 선형 변환 레이어 (reality_stone 통합)
    이론: m x n 가중치 행렬의 모든 m개 행이 하나의 스플라인 위에 놓임
    압축률: (k+1) * n / (m * n) = (k+1) / m
    """
    def __init__(self, in_features: int, out_features: int, 
                 k: int = 50,  # 제어점 개수 - 1
                 use_geodesic: bool = True,
                 bias: bool = True,
                 initial_weight_data: Optional[torch.Tensor] = None,
                 initial_bias_data: Optional[torch.Tensor] = None,
                 module_name_for_debug: str = ""):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.use_geodesic = use_geodesic and RS_AVAILABLE
        self.module_name_for_debug = module_name_for_debug
        
        # 제어점 초기화 - 이론에 맞게 (k+1, in_features) 형태
        if initial_weight_data is not None:
            # 기존 가중치에서 제어점 피팅
            if initial_weight_data.shape != (out_features, in_features):
                if initial_weight_data.shape == (in_features, out_features):
                    initial_weight_data = initial_weight_data.transpose(0, 1)
                else:
                    raise ValueError(f"가중치 형태 불일치: {initial_weight_data.shape}")
            
            self.control_points = nn.Parameter(self._fit_control_points_to_weight(initial_weight_data))
        else:
            # 랜덤 초기화 - (k+1, in_features) 형태
            self.control_points = nn.Parameter(
                torch.randn(k + 1, in_features) * 0.02
            )
        
        # Bias 초기화
        if bias:
            if initial_bias_data is not None:
                if initial_bias_data.shape != (out_features,):
                    raise ValueError(f"Bias 형태 불일치: {initial_bias_data.shape}")
                self.bias = nn.Parameter(initial_bias_data.clone())
            else:
                self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 캐시된 가중치를 위한 버퍼. persistent=False로 state_dict에 저장되지 않도록 함
        self.register_buffer('cached_weight', torch.empty(0), persistent=False)
        
        if self.module_name_for_debug and initial_weight_data is not None:
            with torch.no_grad():
                # .to() 호출을 위해 디바이스를 맞춰줌
                self.control_points = self.control_points.to(initial_weight_data.device)
                if self.bias is not None:
                    self.bias = self.bias.to(initial_weight_data.device)
                reconstructed = self._decompress_weight()
                mse = F.mse_loss(initial_weight_data, reconstructed)
                print(f"    DEBUG [{self.module_name_for_debug}] 스플라인 피팅 MSE: {mse.item():.6f}")

    def _fit_control_points_to_weight(self, target_weight: torch.Tensor) -> nn.Parameter:
        out_features, in_features = target_weight.shape
        # 제어점을 target_weight와 동일한 디바이스에 생성
        temp_control_points = nn.Parameter(
            torch.randn(self.k + 1, in_features, device=target_weight.device) * 0.02
        )
        
        optimizer = torch.optim.AdamW([temp_control_points], lr=1e-2, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        
        progress_bar = tqdm(range(1000), desc=f"피팅 {self.module_name_for_debug}", leave=False)
        
        for step in progress_bar:
            optimizer.zero_grad()
            reconstructed = self._interpolate_from_control_points(temp_control_points)
            loss = F.mse_loss(reconstructed, target_weight)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if step % 100 == 0 or step == 999:
            progress_bar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        progress_bar.close()
        return temp_control_points.detach()

    def _interpolate_from_control_points(self, control_points: torch.Tensor) -> torch.Tensor:
        m = self.out_features
        t_values = torch.linspace(0, 1, m, device=control_points.device)
            
            if self.use_geodesic:
            return geodesic_spline_with_reality_stone(control_points, t_values)
            else:
            return catmull_rom_interpolation(control_points, t_values)

    def _decompress_weight(self) -> torch.Tensor:
        # 평가 모드이고 캐시가 유효하면 캐시된 가중치 사용
        if not self.training and hasattr(self, 'cached_weight') and self.cached_weight.numel() > 0:
            return self.cached_weight

        weight = self._interpolate_from_control_points(self.control_points)
        
        # 평가 모드에서는 가중치를 캐시
        if not self.training:
            self.cached_weight = weight.detach()
        
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        weight = self._decompress_weight()
        # Conv1D 가중치 형식(out, in, 1)에 맞춰 unsqueeze 하고,
        # F.linear 대신 F.conv1d를 사용할 수 있지만,
        # 여기서는 SplineLinearTransform이 Linear와 Conv1D를 모두 대체하므로
        # F.linear를 사용하는 것이 더 일반적임.
        # Conv1D의 weight는 (in_channels, out_channels, width)지만
        # transformers의 Conv1D는 (nf, nx) -> (in_features, out_features) 형태이므로
        # transpose가 필요했었음. SplineLinearTransform은 (out_features, in_features)를 생성하므로
        # F.linear와 완벽하게 호환됨.
        return F.linear(x, weight, self.bias)

    def train(self, mode: bool = True):
        # train 모드로 변경 시 캐시를 비워 다음 eval때 새로 생성하도록 함
        if mode:
            self.cached_weight = torch.empty(0, device=self.control_points.device)
        return super().train(mode)

def convert_linear_to_spline(
    model: nn.Module,
    k: int = 50,
    use_geodesic: bool = True
) -> (nn.Module, Dict[str, Any]):
    """
    모델의 모든 nn.Linear와 Conv1D 레이어를 SplineLinearTransform으로 재귀 없이 교체합니다.
    """
    stats_accumulator = {
            'total_original_params': 0,
            'total_compressed_params': 0,
            'num_layers_converted': 0,
            'conversion_details': []
        }

    modules_to_replace = []
    for name, module in model.named_modules():
        is_linear = isinstance(module, nn.Linear)
        is_conv1d = HF_AVAILABLE and Conv1D is not None and isinstance(module, Conv1D)
        
        # 하위 SplineLinearTransform 레이어는 건너뜀
        if any(isinstance(parent, SplineLinearTransform) for parent in name.split('.')):
            continue
            
        if is_linear or is_conv1d:
            modules_to_replace.append((name, module))

    for name, module in modules_to_replace:
        current_full_path = name
        
        # 부모 모듈과 현재 모듈의 이름을 찾음
        path_tokens = name.split('.')
        parent_module = model
        if len(path_tokens) > 1:
            parent_module = model.get_submodule('.'.join(path_tokens[:-1]))
        child_name = path_tokens[-1]

        # 파라미터 및 가중치 추출
        if isinstance(module, nn.Linear):
            original_params = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
            initial_weight_data = module.weight.data
            in_features, out_features = module.in_features, module.out_features
            shape_info = (out_features, in_features)
            layer_type_info = "Linear"
        else: # Conv1D
            original_params = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
            # transformers Conv1D 가중치는 (in_features, out_features) 형태이므로,
            # (out_features, in_features)로 transpose 필요
            initial_weight_data = module.weight.data.transpose(0, 1)
            in_features, out_features = module.weight.shape[0], module.weight.shape[1]
            shape_info = (out_features, in_features)
            layer_type_info = "Conv1D"

        # Spline 레이어 생성
            spline_layer = SplineLinearTransform(
            in_features=in_features,
            out_features=out_features,
                k=k,
                use_geodesic=use_geodesic,
                bias=module.bias is not None,
            initial_weight_data=initial_weight_data,
                initial_bias_data=module.bias.data if module.bias is not None else None,
                module_name_for_debug=current_full_path
            )
        
        # 모듈 교체
        setattr(parent_module, child_name, spline_layer)

        # 통계 업데이트
            compressed_params = spline_layer.control_points.numel()
            if spline_layer.bias is not None:
                compressed_params += spline_layer.bias.numel()
        stats_accumulator['total_original_params'] += original_params
        stats_accumulator['total_compressed_params'] += compressed_params
        stats_accumulator['num_layers_converted'] += 1
            
        compression_ratio = compressed_params / original_params if original_params > 0 else 0
        stats_accumulator['conversion_details'].append({
                'layer_name': current_full_path,
                'original_params': original_params,
                'compressed_params': compressed_params,
                'compression_ratio': compression_ratio,
            'shape': shape_info
            })
        
        print(f"  ✅ {current_full_path} ({layer_type_info}): {shape_info[0]}×{shape_info[1]} → {k+1} 제어점 (압축률: {compression_ratio:.3f})")

        print(f"\n📊 스플라인 압축 완료:")
    print(f"  변환된 레이어 수: {stats_accumulator['num_layers_converted']}")
    print(f"  원본 파라미터: {stats_accumulator['total_original_params']:,}")
    print(f"  압축 파라미터: {stats_accumulator['total_compressed_params']:,}")
    if stats_accumulator['total_original_params'] > 0:
        overall_compression = stats_accumulator['total_compressed_params'] / stats_accumulator['total_original_params']
        print(f"  전체 압축률: {overall_compression:.3f} ({overall_compression*100:.1f}%)")
        print(f"  메모리 절약: {(1-overall_compression)*100:.1f}%")
    
    return model, stats_accumulator

def get_model_size_mb(model, count_buffers=True):
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    if count_buffers:
        for buffer in model.buffers():
            if buffer is not None:
                buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

@torch.no_grad()
def evaluate_perplexity(model, tokenizer, device, num_samples=100):
    """모델의 Perplexity를 평가합니다."""
    model.eval()
    try:
        # 평가용 데이터셋 로드 (스트리밍으로 빠르게)
        dataset = load_dataset("wikipedia", "20220301.ko", split="train", streaming=True)
        dataset = dataset.take(num_samples)
        texts = [example['text'] for example in dataset if len(example['text']) > 50]
    except Exception as e:
        print(f"Perplexity 평가 데이터셋 로드 실패: {e}")
        return float('inf')

    total_loss = 0
    total_tokens = 0
    
    print(f"\n🤔 Perplexity 평가 중 ({len(texts)}개 샘플)...")
    for text in tqdm(texts, desc="Perplexity 계산", leave=False):
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        input_ids = inputs.input_ids
        
        if input_ids.size(1) < 2:
            continue

        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss * (input_ids.size(1) - 1) # 토큰 수 만큼 loss 곱하기
        
        total_loss += loss.item()
        total_tokens += (input_ids.size(1) - 1)

    if total_tokens == 0:
        return float('inf')
        
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    print(f"  - 평균 Loss: {avg_loss:.4f}")
    print(f"  - Perplexity: {perplexity:.4f}")
    return perplexity

def demo_spline_compression():
    if not HF_AVAILABLE:
        print("\n⚠️ 이 데모를 실행하려면 'transformers'와 'tokenizers' 라이브러리가 필요합니다.")
        print("pip install transformers tokenizers")
        return
    print(f"Reality Stone 사용 가능: {RS_AVAILABLE}")
    model_name = "skt/kogpt2-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")
    original_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    original_model.eval()
    original_params = sum(p.numel() for p in original_model.parameters())
    original_size_mb = get_model_size_mb(original_model)
    
    # 원본 모델 Perplexity 평가
    original_perplexity = evaluate_perplexity(original_model, tokenizer, device)
    
    print(f"\n📊 원본 모델 ({model_name}):")
    print(f"  파라미터 수: {original_params:,}")
    print(f"  모델 크기: {original_size_mb:.2f} MB")
    
    prompt = "인공지능이 세상을 지배하는 시대,"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        original_output_ids = original_model.generate(
            input_ids, max_length=50, num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id
        )
    original_text = tokenizer.decode(original_output_ids[0], skip_special_tokens=True)
    print("\n📝 원본 모델 생성 텍스트:")
    print(original_text)
    print(f"  - Perplexity: {original_perplexity:.4f}")
    k_values = [10, 15, 20, 25] 
    results_summary = []

    for k in k_values:
        print(f"\n" + "="*60)
        print(f"🔧 k={k}로 스플라인 압축 + 파인튜닝 실험")
        test_model = AutoModelForCausalLM.from_pretrained(model_name)
        test_model.load_state_dict(original_model.state_dict())
        test_model.to(device)
        print(f"\n1️⃣ 스플라인 압축 적용 중 (k={k})...")
        compressed_model, stats = convert_linear_to_spline(
            test_model,
            k=k,
            use_geodesic=RS_AVAILABLE
        )
        compressed_model.eval()
        compressed_perplexity = evaluate_perplexity(compressed_model, tokenizer, device)

        with torch.no_grad():
            try:
                compressed_output_ids = compressed_model.generate(
                    input_ids, max_length=50, num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id
                )
                compressed_text_before = tokenizer.decode(compressed_output_ids[0], skip_special_tokens=True)
            except Exception as e:
                compressed_text_before = f"텍스트 생성 실패: {e}"
        print("\n📝 압축 직후 생성 텍스트:")
        print(compressed_text_before)
        print(f"  - Perplexity: {compressed_perplexity:.4f}")

        print(f"\n2️⃣ 압축 모델 파인튜닝 시작...")
        finetuned_model = finetune_compressed_model(
            compressed_model, 
            tokenizer, 
            device=device,
            num_steps = 5000,  
            learning_rate=5e-5
        )
        finetuned_model.eval()
        finetuned_perplexity = evaluate_perplexity(finetuned_model, tokenizer, device)
        with torch.no_grad():
            try:
                finetuned_output_ids = finetuned_model.generate(
                    input_ids, max_length=50, num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id
                )
                finetuned_text = tokenizer.decode(finetuned_output_ids[0], skip_special_tokens=True)
            except Exception as e:
                finetuned_text = f"텍스트 생성 실패: {e}"
        total_compressed_params = sum(p.numel() for p in finetuned_model.parameters())
        compressed_size_mb = get_model_size_mb(finetuned_model, count_buffers=False)
        full_compressed_size_mb = get_model_size_mb(finetuned_model, count_buffers=True)
        compression_ratio = total_compressed_params / original_params
        print(f"\n📈 k={k} 최종 결과:")
        print(f"  압축 모델 총 파라미터: {total_compressed_params:,} (원본의 {compression_ratio*100:.2f}%)")
        print(f"    - 변환된 레이어: {stats['total_compressed_params']:,}개")
        print(f"    - 고정 레이어 (임베딩 등): {total_compressed_params - stats['total_compressed_params']:,}개")
        print(f"  압축 모델 크기 (저장 시): {compressed_size_mb:.2f} MB (원본 대비 {100 - (compressed_size_mb / original_size_mb * 100):.1f}% 감소)")
        print(f"  압축 모델 크기 (버퍼 포함): {full_compressed_size_mb:.2f} MB (원본 대비 {100 - (full_compressed_size_mb / original_size_mb * 100):.1f}% 감소)")
        print("\n📝 파인튜닝 후 생성 텍스트:")
        print(finetuned_text)
        print(f"  - Perplexity: {finetuned_perplexity:.4f}")
        print(f"\n📊 텍스트 품질 비교 (k={k}):")
        print(f"  원본 (PPL: {original_perplexity:.2f}):".ljust(25) + f"{original_text}")
        print(f"  압축 직후 (PPL: {compressed_perplexity:.2f}):".ljust(25) + f"{compressed_text_before}")
        print(f"  파인튜닝 후 (PPL: {finetuned_perplexity:.2f}):".ljust(25) + f"{finetuned_text}")
        
        results_summary.append({
            'k': k,
            'compression_ratio': compression_ratio,
            'compressed_ppl': compressed_perplexity,
            'finetuned_ppl': finetuned_perplexity
        })
    
    print("\n" + "="*60)
    print("📈 최종 요약")
    print("="*60)
    print(f"원본 모델 Perplexity: {original_perplexity:.4f}")
    print("-" * 60)
    print(f"{'k':<5} | {'압축률':<10} | {'압축 후 PPL':<15} | {'파인튜닝 후 PPL':<15}")
    print("-" * 60)
    for res in results_summary:
        print(f"{res['k']:<5} | {res['compression_ratio']:.3f}      | {res['compressed_ppl']:<15.2f} | {res['finetuned_ppl']:<15.2f}")
    print("-" * 60)
        
    return finetuned_model

def prepare_korean_dataset(tokenizer, max_length=512, num_samples=10000):
    """한국어 데이터셋 준비"""
    print("📚 한국어 데이터셋 로드 중...")
    try:
        dataset = load_dataset("wikipedia", "20220301.ko", split="train", streaming=True)
        texts = []
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
            text = example["text"]
            if len(text.strip()) > 50:  
                texts.append(text.strip())
        print(f"  수집된 텍스트: {len(texts)}개")
        
    except Exception as e:
        print(f"  위키피디아 로드 실패: {e}")
        raise e
    print("  토크나이징 중...")
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors="pt"
    )
    class SimpleDataset:
        def __init__(self, input_ids):
            self.input_ids = input_ids
        
        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[idx]}
    
    dataset = SimpleDataset(tokenized["input_ids"])
    print(f"  데이터셋 준비 완료: {len(dataset)}개 샘플")
    
    return dataset

def finetune_compressed_model(model, tokenizer, device, num_steps=50000, learning_rate=1e-4):
    """압축된 모델 파인튜닝"""
    print(f"🔧 압축 모델 파인튜닝 시작 ({num_steps:,} 스텝)")
    # 모델을 올바른 디바이스로 이동
    model.to(device)
    model.train()

    train_dataset = prepare_korean_dataset(tokenizer, num_samples=5000)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
        return_tensors="pt"
    )
    training_args = TrainingArguments(
        output_dir="./spline_finetuned",
        overwrite_output_dir=True,
        max_steps=num_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        warmup_steps=1000,
        logging_steps=1000,
        save_steps=10000,
        evaluation_strategy="no",
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_drop_last=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    print("  파인튜닝 시작...")
    trainer.train()
    print("✅ 파인튜닝 완료!")
    return model

if __name__ == "__main__":
    demo_spline_compression() 