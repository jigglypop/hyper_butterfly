import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Union
import reality_stone as rs # reality_stone 라이브러리가 설치되어 있다고 가정
from transformers import AutoTokenizer, AutoModelForCausalLM # 한국어 생성 테스트용
from transformers.modeling_utils import Conv1D # GPT-2 MLP 레이어 처리를 위해 추가

class RiemannLinearTransform(nn.Module):
    def __init__(self, in_features: int, out_features: int, curvature: float = 1.0,
                 svd_rank_or_ratio: Optional[Union[int, float]] = None, 
                 fft_compression_ratio: float = 0.999, # Default for nearly lossless for FFT part
                 bias: bool = True,
                 initial_weight_data: Optional[torch.Tensor] = None,
                 initial_bias_data: Optional[torch.Tensor] = None,
                 activation_type: Optional[str] = None,
                 module_name_for_debug: str = ""):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.curvature = curvature
        self.svd_rank_or_ratio = svd_rank_or_ratio
        self.fft_compression_ratio = fft_compression_ratio
        self.activation_type = activation_type
        self.module_name_for_debug = module_name_for_debug

        # Determine SVD non-compression condition
        svd_is_full_rank = False
        if self.svd_rank_or_ratio is None:
            svd_is_full_rank = True
        elif isinstance(self.svd_rank_or_ratio, float) and self.svd_rank_or_ratio >= 0.999:
            svd_is_full_rank = True
        elif isinstance(self.svd_rank_or_ratio, int) and self.svd_rank_or_ratio >= min(self.in_features, self.out_features):
            svd_is_full_rank = True
        
        # Determine FFT non-compression condition
        fft_is_full_coeffs = self.fft_compression_ratio >= 0.999

        self.is_effectively_lossless = svd_is_full_rank and fft_is_full_coeffs
        
        # Debug print for the flag
        # if self.module_name_for_debug:
        #     print(f"    DEBUG [{self.module_name_for_debug}]: svd_full={svd_is_full_rank} (ratio/rank={self.svd_rank_or_ratio}), fft_full={fft_is_full_coeffs} (ratio={self.fft_compression_ratio}) ==> is_lossless={self.is_effectively_lossless}")

        if initial_weight_data is not None:
            if initial_weight_data.shape != (out_features, in_features):
                if initial_weight_data.shape == (in_features, out_features):
                    initial_weight_data = initial_weight_data.transpose(0,1)
                else:
                    raise ValueError(f"Initial weight shape mismatch for RLT {self.module_name_for_debug}. Expected {(out_features, in_features)} or {(in_features, out_features)}, got {initial_weight_data.shape}")
            original_device = initial_weight_data.device

            if self.is_effectively_lossless:
                # print(f"    DEBUG [{self.module_name_for_debug}]: Using lossless path (direct weight_param).")
                self.weight_param = nn.Parameter(initial_weight_data.clone())
                self.U_real, self.U_imag, self.S_param, self.Vt_real, self.Vt_imag = [None]*5
                self.U_coeffs_stored, self.Vt_coeffs_stored, self.k_rank = [0]*3
            else:
                # print(f"    DEBUG [{self.module_name_for_debug}]: Using SVD/FFT compression path.")
                self.register_parameter('weight_param', None)
                U, S, Vh = torch.linalg.svd(initial_weight_data, full_matrices=False)
                min_dim = min(self.out_features, self.in_features)

                if svd_is_full_rank or self.svd_rank_or_ratio is None: # No SVD compression or svd_rank_or_ratio implies full
                    self.k_rank = min_dim
                elif isinstance(self.svd_rank_or_ratio, float):
                    self.k_rank = max(1, int(min_dim * self.svd_rank_or_ratio))
                elif isinstance(self.svd_rank_or_ratio, int):
                    self.k_rank = max(1, self.svd_rank_or_ratio)
                self.k_rank = min(self.k_rank, min_dim)

                U_k = U[:, :self.k_rank]
                S_k = S[:self.k_rank]
                Vh_k = Vh[:self.k_rank, :]
                self.S_param = nn.Parameter(S_k.clone())

                def _compress_matrix_fft(matrix: torch.Tensor, ratio: float, original_fft_dim_len: int, name_prefix: str) -> Tuple[nn.Parameter, nn.Parameter, int]:
                    matrix_rfft = torch.fft.rfft(matrix, n=original_fft_dim_len, dim=-1)
                    num_unique_coeffs = matrix_rfft.shape[-1]
                    num_coeffs_to_store = num_unique_coeffs if ratio >= 0.999 else max(1, int(num_unique_coeffs * ratio))
                    num_coeffs_to_store = min(num_coeffs_to_store, num_unique_coeffs)
                    coeffs_to_store = matrix_rfft[..., :num_coeffs_to_store]
                    return nn.Parameter(coeffs_to_store.real.clone()), nn.Parameter(coeffs_to_store.imag.clone()), num_coeffs_to_store

                # FFT for U_k is on its last dim (k_rank)
                # FFT for Vh_k is on its last dim (in_features)
                self.U_real, self.U_imag, self.U_coeffs_stored = _compress_matrix_fft(U_k, self.fft_compression_ratio, self.k_rank, f"{self.module_name_for_debug}_U_k")
                self.Vt_real, self.Vt_imag, self.Vt_coeffs_stored = _compress_matrix_fft(Vh_k, self.fft_compression_ratio, self.in_features, f"{self.module_name_for_debug}_Vh_k")
                
                if self.module_name_for_debug: # Debug print for lossy path only
                    with torch.no_grad():
                        decompressed_at_init = self._decompress_weight().to(original_device)
                        mse = F.mse_loss(initial_weight_data, decompressed_at_init)
                        print(f"    DEBUG [{self.module_name_for_debug} SVD(k={self.k_rank})+FFT(r={self.fft_compression_ratio:.3f}) init]: InitialW vs DecompW MSE: {mse.item():.4e}")
        else: # No initial_weight_data (train from scratch)
            if self.is_effectively_lossless:
                self.weight_param = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
                self.U_real, self.U_imag, self.S_param, self.Vt_real, self.Vt_imag = [None]*5
                self.U_coeffs_stored, self.Vt_coeffs_stored, self.k_rank = [0]*3
            else:
                self.register_parameter('weight_param', None)
                min_dim = min(self.out_features, self.in_features)
                if self.svd_rank_or_ratio is None or (isinstance(self.svd_rank_or_ratio, float) and self.svd_rank_or_ratio >=0.999) or (isinstance(self.svd_rank_or_ratio, int) and self.svd_rank_or_ratio >= min_dim) :
                     self.k_rank = min_dim
                elif isinstance(self.svd_rank_or_ratio, float):
                    self.k_rank = max(1, int(min_dim * self.svd_rank_or_ratio))
                else: # int
                    self.k_rank = max(1, self.svd_rank_or_ratio)
                self.k_rank = min(self.k_rank, min_dim)
                self.S_param = nn.Parameter(torch.randn(self.k_rank) * 0.02)
                def _init_compressed_fft_params(dim1_size: int, original_fft_dim_len: int, ratio: float) -> Tuple[nn.Parameter, nn.Parameter, int]:
                    num_unique_coeffs = original_fft_dim_len // 2 + 1
                    num_coeffs_to_store = num_unique_coeffs if ratio >=0.999 else max(1, int(num_unique_coeffs * ratio))
                    num_coeffs_to_store = min(num_coeffs_to_store, num_unique_coeffs)
                    return nn.Parameter(torch.randn(dim1_size, num_coeffs_to_store) * 0.01), \
                           nn.Parameter(torch.randn(dim1_size, num_coeffs_to_store) * 0.01), \
                           num_coeffs_to_store
                self.U_real, self.U_imag, self.U_coeffs_stored = _init_compressed_fft_params(self.out_features, self.k_rank, self.fft_compression_ratio)
                self.Vt_real, self.Vt_imag, self.Vt_coeffs_stored = _init_compressed_fft_params(self.k_rank, self.in_features, self.fft_compression_ratio)

        # Bias and activation filter parameters (common to both modes)
        if bias:
            if initial_bias_data is not None:
                if initial_bias_data.shape != (out_features,):
                    raise ValueError(f"Initial bias shape mismatch for RLT {self.module_name_for_debug}. Expected {(out_features,)}, got {initial_bias_data.shape}")
                self.bias = nn.Parameter(initial_bias_data.clone())
            else:
                self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Activation filter parameters can be smaller if we are in lossless mode for weights
        # For now, their size still depends on fft_compression_ratio for activation transformation (which is currently identity)
        max_freq_for_activation = min(self.out_features, 128) 
        # If effectively lossless for weights, maybe activation filter should also be minimal or disabled?
        # Current logic: if fft_compression_ratio is high, activation filter params are also larger.
        active_filter_ratio = self.fft_compression_ratio # Link to main FFT ratio, or make it independent
        compressed_dim_for_activation = max(1, int(max_freq_for_activation * active_filter_ratio)) 
        self.activation_complex_filter_real = nn.Parameter(torch.randn(compressed_dim_for_activation) * 0.01)
        self.activation_complex_filter_imag = nn.Parameter(torch.randn(compressed_dim_for_activation) * 0.01)
        self.hyp_relu_threshold = nn.Parameter(torch.tensor(0.0))
        self.hyp_relu_slope = nn.Parameter(torch.tensor(0.1))

    def _decompress_weight(self) -> torch.Tensor:
        if self.is_effectively_lossless and self.weight_param is not None:
            return self.weight_param
        
        # This path is for when SVD and/or FFT compression is active
        if self.S_param is None or self.U_real is None or self.U_imag is None or self.Vt_real is None or self.Vt_imag is None:
             if self.weight_param is not None: # Should only happen if somehow is_effectively_lossless was false but components are missing
                 # This case implies an issue in __init__ logic for non-lossless without initial_weight_data
                 # or if is_effectively_lossless was miscalculated and this path was taken incorrectly.
                 print(f"Warning: [{self.module_name_for_debug}] SVD/FFT components missing, but not in lossless mode. Falling back to weight_param if available.")
                 return self.weight_param
             raise RuntimeError(f"[{self.module_name_for_debug}] SVD/FFT components not properly initialized for decompression in lossy mode.")

        def _decompress_matrix_component(coeffs_real: nn.Parameter, coeffs_imag: nn.Parameter, 
                                         num_coeffs_stored: int, original_fft_target_dim_len: int, 
                                         component_name_for_debug:str) -> torch.Tensor:
            coeffs_complex = torch.complex(coeffs_real, coeffs_imag)
            # For rfft, the number of unique complex coefficients for a real signal of length N is N//2 + 1
            num_unique_coeffs_original = original_fft_target_dim_len // 2 + 1
            
            # print(f"    DEBUG Decompressing {component_name_for_debug}: stored_coeffs={num_coeffs_stored}, target_fft_len={original_fft_target_dim_len}, unique_coeffs_for_target={num_unique_coeffs_original}")
            # print(f"    DEBUG Coeffs_complex shape: {coeffs_complex.shape}")

            if num_coeffs_stored < num_unique_coeffs_original:
                padding_size = num_unique_coeffs_original - num_coeffs_stored
                padded_coeffs = F.pad(coeffs_complex, (0, padding_size), "constant", 0)
            elif num_coeffs_stored == num_unique_coeffs_original:
                padded_coeffs = coeffs_complex
            else: # num_coeffs_stored > num_unique_coeffs_original, should not happen if _compress_matrix_fft is correct
                # print(f"    WARNING: [{component_name_for_debug}] num_coeffs_stored ({num_coeffs_stored}) > num_unique_coeffs_original ({num_unique_coeffs_original}). Truncating.")
                padded_coeffs = coeffs_complex[..., :num_unique_coeffs_original]
            
            # print(f"    DEBUG Padded_coeffs shape for {component_name_for_debug}: {padded_coeffs.shape}")
            # The `n` parameter for irfft is the length of the original _real_ signal along the transformed axis.
            decomp_matrix = torch.fft.irfft(padded_coeffs, n=original_fft_target_dim_len, dim=-1)
            # print(f"    DEBUG Decompressed matrix {component_name_for_debug} shape: {decomp_matrix.shape}")
            return decomp_matrix

        # U_k original shape before FFT: (self.out_features, self.k_rank)
        # FFT was applied on self.k_rank dimension
        U_k_decomp = _decompress_matrix_component(self.U_real, self.U_imag, self.U_coeffs_stored, 
                                                self.k_rank, f"{self.module_name_for_debug}_U_k")
        
        # Vh_k original shape before FFT: (self.k_rank, self.in_features)
        # FFT was applied on self.in_features dimension
        Vh_k_decomp = _decompress_matrix_component(self.Vt_real, self.Vt_imag, self.Vt_coeffs_stored, 
                                                 self.in_features, f"{self.module_name_for_debug}_Vh_k")
        
        # Expected shapes for matmul: 
        # U_k_decomp: (self.out_features, self.k_rank)
        # S_param: (self.k_rank)
        # Vh_k_decomp: (self.k_rank, self.in_features)
        # Result: (self.out_features, self.in_features)
        
        # print(f"    DEBUG Shapes for SVD reconstruction: U_k_decomp={U_k_decomp.shape}, S_param={self.S_param.shape}, Vh_k_decomp={Vh_k_decomp.shape}")

        # W = U @ diag(S) @ Vh  or  W = (U*S_broadcasted) @ Vh
        try:
            decompressed_weight = (U_k_decomp * self.S_param.unsqueeze(0)) @ Vh_k_decomp
        except RuntimeError as e:
            print(f"Error during SVD reconstruction matmul for {self.module_name_for_debug}: {e}")
            print(f"Shapes: U_k_decomp={U_k_decomp.shape}, S_param={self.S_param.shape}, Vh_k_decomp={Vh_k_decomp.shape}")
            raise e
            
        return decompressed_weight

    def euclidean_to_hyperbolic(self, x: torch.Tensor, C: Optional[float] = None) -> torch.Tensor:
        c_val = C if C is not None else self.curvature
        if c_val <= 0: return x # 곡률 0 또는 음수면 유클리드 공간으로 간주
        norm_x_sq = torch.sum(x * x, dim=-1, keepdim=True)
        # Numerical stability: ensure sqrt(c_val) * ||x|| is not too large for tanh
        # max_norm = 0.99 / (torch.sqrt(torch.tensor(c_val, device=x.device)) + 1e-9)
        # x_clipped = x * torch.clamp(max_norm / (torch.norm(x, dim=-1, keepdim=True) + 1e-9), max=1.0)
        # norm_x_sq_clipped = torch.sum(x_clipped * x_clipped, dim=-1, keepdim=True)

        # Using Poincare ball model: x_h = x / (1 + sqrt(1 + c|x|^2)) - not standard.
        # Standard Poincare exp map from origin: x_h = tanh(sqrt(c)*||x||) * x / (sqrt(c)*||x||)
        norm = torch.sqrt(norm_x_sq + 1e-9)
        sqrt_c = torch.sqrt(torch.tensor(c_val, device=x.device))
        factor = torch.tanh(sqrt_c * norm) / (sqrt_c * norm + 1e-9) # Add epsilon for norm=0 case
        return factor * x
    
    def hyperbolic_to_euclidean(self, x_h: torch.Tensor, C: Optional[float] = None) -> torch.Tensor:
        c_val = C if C is not None else self.curvature
        if c_val <= 0: return x_h
        norm_xh_sq = torch.sum(x_h * x_h, dim=-1, keepdim=True)
        # Ensure c_val * norm_xh_sq < 1 for atanh
        # x_h_clipped = x_h * torch.clamp(0.99 / (torch.sqrt(torch.tensor(c_val, device=x_h.device)) * torch.norm(x_h, dim=-1, keepdim=True) + 1e-9), max=1.0)
        # norm_xh_sq_clipped = torch.sum(x_h_clipped * x_h_clipped, dim=-1, keepdim=True)

        # Standard Poincare log map to origin: x = atanh(sqrt(c)*||x_h||) * x_h / (sqrt(c)*||x_h||)
        norm = torch.sqrt(norm_xh_sq + 1e-9)
        sqrt_c = torch.sqrt(torch.tensor(c_val, device=x_h.device))
        
        # Clamp argument of atanh to be < 1 for stability
        arg_atanh = torch.clamp(sqrt_c * norm, max=1.0 - 1e-7)
        factor = torch.atanh(arg_atanh) / (sqrt_c * norm + 1e-9)
        return factor * x_h
    
    def mobius_linear_transform(self, x_h: torch.Tensor) -> torch.Tensor:
        decompressed_weight = self._decompress_weight()
        
        # 유클리드 공간에서 선형 변환 적용 (하이퍼볼릭 공간으로 매핑된 x_h에 대해)
        # 실제로는 이 변환이 뫼비우스 변환의 일부가 되도록 해야함.
        # Gyrovector spaces formalism: Wx_h + b_h where b_h is also in hyperbolic space
        # Simpler Poincare approach: transform tangent space vector, then map back.
        # Here, we apply linear transform then try to make it Mobius-like.

        # Apply linear transformation in Euclidean space (as if x_h is Euclidean)
        # This is an approximation or a specific type of hyperbolic linear layer.
        # W_euc * log_map(x_h) -> exp_map(...)
        # For now, let's assume x_h is operated on directly, then wrapped by Mobius.
        
        z_euc = F.linear(x_h, decompressed_weight, self.bias)

        if self.curvature <= 0: return z_euc # 유클리드처럼 동작

        # 뫼비우스 일반화 선형 변환 (근사적 또는 특정 형태)
        # For Poincare ball, a common form is M(x) = exp_0 ( A log_0(x) + t_tangent )
        # A simpler, often used one is based on Poincare distance preserving transformations.
        # Or, a direct generalization Wx / (1 + <c_factor*W,x>) - this needs careful derivation for bias.
        # A common simplification: apply Euclidean linear, then project/map.
        # Let's use a common Mobius addition form for bias: z_h = ( (1+2c<x,b>+c|b|^2)x + (1-c|x|^2)b ) / (1+2c<x,b>+c|x|^2c|b|^2)
        # Or simpler: Project Wx+b to the ball.

        # Using a form similar to HyperTorch or other libraries:
        # Map x_h to tangent space at origin, linear transform, map back, then Mobius add bias.
        # x_tangent = self.hyperbolic_to_euclidean(x_h, self.curvature) # log_0(x_h)
        # transformed_tangent = F.linear(x_tangent, decompressed_weight)
        # z_h_no_bias = self.euclidean_to_hyperbolic(transformed_tangent, self.curvature) # exp_0(Wx_tangent)

        # If bias is present, it should be a Mobius addition.
        # For now, let's use a simplified Mobius-like transformation on z_euc
        # This is an approximation and might not preserve all hyperbolic properties perfectly.
        
        if self.bias is not None:
            # Apply Mobius addition for the bias term (bias itself treated as a point in tangent space)
            # b_h = self.euclidean_to_hyperbolic(self.bias.unsqueeze(0), self.curvature) # map bias to hyperbolic space
            # This is complex. A simpler approximation for bias effect:
            # Let z_euc = Wx_h + b.
            # We need to ensure the output is in the Poincare ball.
            # One way is to scale z_euc to fit, but this may not be 'linear' in hyperbolic sense.
            # A common way to introduce bias in hyperbolic layers is to transform x, then add bias (in tangent space), then map back.
            # Or treat bias as a translation: p(+)b = exp_p (log_p(0) + b_tangent)
            # The current z_euc = Wx_h + b (Euclidean)
            # To make it "Mobius-like" and ensure it's in the ball:
            c = self.curvature
            # A more direct application of Mobius transform structure from some papers on Wx+b:
            # (Not rigorously derived here, but a common pattern)
            # This form does not perfectly align with theory without careful weight/bias constraints.
            # Numerically stable projection/scaling as a fallback:
            z_norm_sq = torch.sum(z_euc * z_euc, dim=-1, keepdim=True)
            denominator_factor = c * z_norm_sq
            # if using the form (Wx+b)/(1+c<Wx,b>), it implies bias is also scaled by W somehow.
            # For simplicity and to ensure it's in the ball, let's use a projection if it goes out.
            # Or a learnable scaling factor within Mobius definition if available.
            
            # Fallback: if result of Euclidean linear is outside, project it.
            # This is a practical simplification.
            max_norm_val = 1.0 / (torch.sqrt(torch.tensor(c, device=z_euc.device)) + 1e-7)
            current_norm = torch.norm(z_euc, dim=-1, keepdim=True)
            z_h = z_euc * torch.clamp(max_norm_val / (current_norm + 1e-9), max=1.0)
        else:
            z_h = z_euc # No bias, Wx_h, still needs projection if c > 0

        # Ensure output is in the ball if curvature is positive
        if self.curvature > 0:
             max_norm_val = 1.0 / (torch.sqrt(torch.tensor(self.curvature, device=z_h.device)) + 1e-7)
             current_norm = torch.norm(z_h, dim=-1, keepdim=True)
             z_h = z_h * torch.clamp(max_norm_val / (current_norm + 1e-9), max=1.0)
        
        return z_h

    def hyperbolic_relu(self, z_h: torch.Tensor) -> torch.Tensor:
        if self.curvature <= 0: # Euclidean ReLU
             return F.relu(z_h)

        # 하이퍼볼릭 공간에서의 "양수" 판별: 원점과의 거리 기준 또는 특정 방향 벡터와의 내적
        # 여기서는 간단히 원점과의 거리 사용
        distance_from_origin = torch.norm(z_h, dim=-1, keepdim=True)
        
        # 임계값 기준 활성화
        mask = distance_from_origin > self.hyp_relu_threshold
        
        # 활성화된 부분은 그대로, 비활성화된 부분은 축소 (학습 가능한 slope 적용)
        # 축소 시 방향 보존
        # sigmoid(slope) * 0.1 -> makes the factor small and learnable
        reduction_factor = torch.sigmoid(self.hyp_relu_slope) * 0.1 
        
        # result = torch.where(mask, z_h, z_h * reduction_factor) # Incorrect: where needs broadcastable reduction_factor
        # Corrected:
        active_part = z_h
        inactive_part = z_h * reduction_factor
        
        result = torch.where(mask, active_part, inactive_part)
        return result

    def compressed_spectral_transform_activation(self, x: torch.Tensor) -> torch.Tensor:
        # """활성화 값에 대한 압축된 스펙트럴 변환 (Reality Stone 하이퍼볼릭 FFT 활용)"""
        # 디버깅: 이 변환을 임시로 비활성화하고 입력을 그대로 반환
        return x 
        
        # # 원래 코드 시작
        # use_rs_fft = hasattr(rs, 'hyperbolic_fft') and x.is_cuda and self.curvature > 0
        # try:
        #     x_fft = rs.hyperbolic_fft(x, self.curvature) if use_rs_fft else torch.fft.fft(x.to(torch.complex64), dim=-1)
        #     filter_coeffs_to_use = min(x_fft.shape[-1], self.activation_complex_filter_real.shape[0])
        #     complex_activation_filter = torch.complex(
        #         self.activation_complex_filter_real[:filter_coeffs_to_use],
        #         self.activation_complex_filter_imag[:filter_coeffs_to_use]
        #     ).to(x_fft.device)
        #     if x_fft.shape[-1] > filter_coeffs_to_use:
        #          filtered_part = x_fft[..., :filter_coeffs_to_use] * complex_activation_filter
        #          padding_zeros = torch.zeros_like(x_fft[..., filter_coeffs_to_use:])
        #          x_fft_filtered_padded = torch.cat([filtered_part, padding_zeros], dim=-1)
        #     else:
        #          x_fft_filtered_padded = x_fft * complex_activation_filter[:x_fft.shape[-1]]
        #     if use_rs_fft:
        #         x_compressed = rs.inverse_hyperbolic_fft(x_fft_filtered_padded.real, self.curvature)
        #     else:
        #         x_compressed = torch.fft.ifft(x_fft_filtered_padded, dim=-1).real
        #     return x_compressed
        # except Exception:
        #     return x 
        # # 원래 코드 끝

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        # 단계 2에서 확장될 부분
        if self.activation_type == 'relu':
            if self.curvature > 0:
                # 현재 hyperbolic_relu를 사용하거나, 유클리드 ReLU를 탄젠트 공간에서 적용 후 다시 매핑
                return self.hyperbolic_relu(x) 
            else:
                return F.relu(x)
        elif self.activation_type == 'gelu':
            # GELU의 하이퍼볼릭 버전은 연구가 필요. 일단 유클리드 GELU 적용.
            return F.gelu(x) 
        # 여기에 다른 활성화 함수들 추가 가능
        
        # activation_type이 명시되지 않았거나, 위의 경우에 해당하지 않으면 hyperbolic_relu를 기본으로 사용
        # (또는 아무것도 안하거나, curvature > 0 일때만 hyperbolic_relu를 사용하도록 수정)
        # 기존 로직은 hyperbolic_relu를 forward에서 직접 호출했었음.
        # 여기서는 activation_type에 따라 분기하고, 없으면 (또는 'hyp_relu' 명시 시) hyperbolic_relu 호출
        if self.activation_type == 'hyp_relu' or (self.activation_type is None and self.curvature > 0):
             return self.hyperbolic_relu(x)
        elif self.activation_type is None and self.curvature <= 0: # No activation type, Euclidean
             return x # No activation, pass through
        
        # If activation type specified but not handled above, default to pass-through or error
        # For now, let's default to hyperbolic_relu if curvature > 0, else pass-through if no type matches
        # This part needs careful decision based on desired default behavior.
        # The logic above tries to cover common cases.
        return self.hyperbolic_relu(x) if self.curvature > 0 else x # Simplified default

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 유클리드 → 하이퍼볼릭 공간 (입력에 대해)
        x_h_input = self.euclidean_to_hyperbolic(x)

        # 2. 하이퍼볼릭 공간에서 뫼비우스 선형변환 (압축 해제된 가중치 사용)
        z_h = self.mobius_linear_transform(x_h_input)

        # 3. 하이퍼볼릭 ReLU 활성화
        a_h = self._apply_activation(z_h)

        # 4. (선택적) 활성화 값에 대한 압축된 스펙트럴 변환
        #   이 단계는 추가적인 비선형성 또는 정보 압축/변형을 제공할 수 있음
        #   만약 이 단계의 출력이 다시 하이퍼볼릭 공간에 있어야 한다면, 추가적인 매핑 필요
        #   현재 compressed_spectral_transform_activation은 유클리드 공간 출력을 가정
        s_euc = self.compressed_spectral_transform_activation(a_h) 
        
        # 5. 다음 레이어를 위해 유클리드 공간으로 변환할지, 하이퍼볼릭으로 유지할지 결정.
        #    여기서는 최종적으로 유클리드 출력을 가정.
        #    만약 다음 레이어도 RiemannLinearTransform이면, 하이퍼볼릭으로 전달 가능.
        #    일단 유클리드로 변환하여 일반적인 nn.Module과 호환되도록 함.
        y = self.hyperbolic_to_euclidean(s_euc if self.curvature > 0 and torch.is_tensor(s_euc) and s_euc.numel() > 0 and not torch.all(s_euc == 0) else a_h) # s_euc가 유효한 값이면 사용, 아니면 a_h 사용

        return y

def helgason_fuse_sequential_linear_layers(model: nn.Module, verbose: bool = False) -> nn.Module:
    """
    (New Helgason Fusion Function - Phase 1)
    Fuses sequences of purely linear layers (nn.Linear or Conv1D kernel=1)
    into single equivalent nn.Linear layers. This is a lossless transformation.
    This function modifies the model in-place.
    """
    if verbose:
        print("🔥 Helgason Linear Fusion Pass Starting...")
    
    for name, module in list(model.named_children()): # Iterate over a copy for in-place modification
        if isinstance(module, nn.Sequential):
            if verbose:
                print(f"  Scanning nn.Sequential: {name}")
            new_sequential_children = []
            current_linear_sequence = []
            
            for i, layer in enumerate(module):
                is_linear_type = isinstance(layer, nn.Linear) or \
                                 (isinstance(layer, Conv1D) and layer.weight.shape[2:] == (1,1) if len(layer.weight.shape) == 4 else layer.weight.shape[2:] == (1,) if len(layer.weight.shape) == 3 else True) # kernel_size=1 check
                
                if is_linear_type:
                    current_linear_sequence.append(layer)
                else:
                    if len(current_linear_sequence) > 1:
                        # Fuse the collected sequence
                        if verbose: print(f"    Found sequence of {len(current_linear_sequence)} linear layers to fuse in {name}.")
                        fused_layer = _fuse_linear_block(current_linear_sequence, verbose)
                        new_sequential_children.append(fused_layer)
                    elif len(current_linear_sequence) == 1:
                        new_sequential_children.append(current_linear_sequence[0])
                    current_linear_sequence = []
                    new_sequential_children.append(layer) # Add the non-linear layer
            
            # After loop, process any remaining sequence
            if len(current_linear_sequence) > 1:
                if verbose: print(f"    Found sequence of {len(current_linear_sequence)} linear layers to fuse at the end of {name}.")
                fused_layer = _fuse_linear_block(current_linear_sequence, verbose)
                new_sequential_children.append(fused_layer)
            elif len(current_linear_sequence) == 1:
                new_sequential_children.append(current_linear_sequence[0])
            
            # Replace the old sequential module with the new one containing fused layers
            if len(new_sequential_children) != len(module):
                 if verbose: print(f"    Rebuilding Sequential module '{name}' with fused layers.")
                 model._modules[name] = nn.Sequential(*new_sequential_children)
            elif verbose:
                 print(f"    No fusion occurred in Sequential module '{name}'.")

        elif len(list(module.children())) > 0: # If not Sequential, recurse for other container types
            helgason_fuse_sequential_linear_layers(module, verbose=verbose)
            
    if verbose:
        print("🔥 Helgason Linear Fusion Pass Complete.")
    return model

def _fuse_linear_block(layers: List[nn.Module], verbose: bool = False) -> nn.Linear:
    """Helper to fuse a list of nn.Linear or Conv1D (kernel 1) layers."""
    # Ensure all layers are compatible (Linear or Conv1D acting as Linear)
    # For simplicity, this example assumes weights are 2D or can be treated as such.
    # A more robust implementation would handle Conv1D shapes meticulously.

    current_weight: Optional[torch.Tensor] = None
    current_bias: Optional[torch.Tensor] = None

    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            L_weight = layer.weight.data.clone()
            L_bias = layer.bias.data.clone() if layer.bias is not None else None
        elif isinstance(layer, Conv1D): # Assuming Conv1D here is like transformers.modeling_utils.Conv1D
            # Conv1D weight: (in_channels, out_channels) in HF, but kernel_size=1 means it acts like Linear(nx, nf)
            # weight shape: (nx, nf), bias shape: (nf,)
            # To match nn.Linear (out_features, in_features), we need W.T
            L_weight = layer.weight.data.T.clone() # (nf, nx) - out_features, in_features
            L_bias = layer.bias.data.clone() if layer.bias is not None else None
        else:
            raise TypeError(f"Layer type {type(layer)} not supported for Helgason fusion in this simple version.")

        if current_weight is None: # First layer in sequence
            current_weight = L_weight
            current_bias = L_bias
        else:
            # W_eff = W_new @ W_old
            current_weight = torch.matmul(L_weight, current_weight)
            # b_eff = W_new @ b_old + b_new
            if current_bias is not None:
                new_bias_part = torch.matmul(L_weight, current_bias.unsqueeze(-1)).squeeze(-1)
            else:
                new_bias_part = torch.zeros_like(L_bias) if L_bias is not None else 0.0
            
            if L_bias is not None:
                current_bias = new_bias_part + L_bias
            elif isinstance(new_bias_part, torch.Tensor): # L_bias was None but new_bias_part is tensor
                current_bias = new_bias_part
            else: # Both L_bias and new_bias_part are effectively None/0
                current_bias = None
    
    if current_weight is None:
        raise ValueError("Cannot fuse an empty list of layers.")

    fused_linear = nn.Linear(current_weight.shape[1], current_weight.shape[0],
                               bias=(current_bias is not None))
    fused_linear.weight.data = current_weight
    if current_bias is not None:
        fused_linear.bias.data = current_bias
    
    if verbose:
        print(f"      Fused {len(layers)} layers into one nn.Linear({current_weight.shape[1]}, {current_weight.shape[0]})")
    return fused_linear

def convert_linear_to_riemann(
    model: nn.Module,
    curvature: float = 1.0,
    svd_rank_or_ratio: Optional[Union[int, float]] = None, 
    fft_compression_ratio: float = 0.999, 
    _current_path: Optional[List[str]] = None,
    _processed_names: Optional[set] = None,
    _stats_accumulator: Optional[Dict[str, Any]] = None
) -> nn.Module:
    is_top_level_call = _stats_accumulator is None
    if is_top_level_call:
        print(f"🧮 Riemann (SVD+FFT) 변환 시작 (SVD: {svd_rank_or_ratio}, FFT ratio: {fft_compression_ratio:.3f})")
        _current_path = []
        _processed_names = set()
        _stats_accumulator = {
            "replaced_count": 0,
            "initial_total_params": sum(p.numel() for p in model.parameters()),
            "current_total_params_after_conversion": sum(p.numel() for p in model.parameters())
        }
    
    children_names = [name for name, _ in model.named_children()]
    idx = 0
    while idx < len(children_names):
        name = children_names[idx]
        child_module = getattr(model, name)
        path_key_list = _current_path + [name]
        full_module_name_tuple = tuple(path_key_list)
        full_module_name_str = ".".join(path_key_list)

        if full_module_name_tuple in _processed_names:
            idx += 1
            continue

        if name == 'lm_head' or full_module_name_str.endswith('.lm_head'):
            _processed_names.add(full_module_name_tuple)
            if len(list(child_module.children())) > 0:
                convert_linear_to_riemann(child_module, curvature, svd_rank_or_ratio, fft_compression_ratio, 
                                        _current_path=path_key_list, 
                                        _processed_names=_processed_names, 
                                        _stats_accumulator=_stats_accumulator)
            idx += 1
            continue
        
        target_layer_module = None; initial_weight = None; initial_bias = None
        in_feats, out_feats = 0, 0; original_type_name = ""
        params_in_original_target_layer = 0

        if isinstance(child_module, nn.Linear):
            target_layer_module = child_module
            params_in_original_target_layer = sum(p.numel() for p in target_layer_module.parameters())
            in_feats, out_feats = target_layer_module.in_features, target_layer_module.out_features
            initial_weight = target_layer_module.weight.data.clone()
            initial_bias = target_layer_module.bias.data.clone() if target_layer_module.bias is not None else None
            original_type_name = "Linear"
        elif isinstance(child_module, Conv1D):
            target_layer_module = child_module
            params_in_original_target_layer = sum(p.numel() for p in target_layer_module.parameters())
            nx, nf = target_layer_module.weight.shape[0], target_layer_module.weight.shape[1]
            out_feats = nf; in_feats  = nx
            initial_weight = target_layer_module.weight.data.transpose(0,1).clone()
            initial_bias = target_layer_module.bias.data.clone() if hasattr(target_layer_module, 'bias') and target_layer_module.bias is not None else None
            original_type_name = "Conv1D"

        if target_layer_module:
            _processed_names.add(full_module_name_tuple)
            activation_to_fuse_type_str = None; activation_module_original_name = None
            params_in_fused_activation = 0
            if idx + 1 < len(children_names):
                next_module_name_candidate = children_names[idx+1]
                next_module_candidate = getattr(model, next_module_name_candidate)
                temp_act_type_str = ""
                if isinstance(next_module_candidate, (nn.ReLU, nn.GELU)):
                    temp_act_type_str = type(next_module_candidate).__name__.lower().replace("activation","")
                if temp_act_type_str:
                    activation_to_fuse_type_str = temp_act_type_str
                    activation_module_original_name = next_module_name_candidate
                    params_in_fused_activation = sum(p.numel() for p in next_module_candidate.parameters())
                    _processed_names.add(tuple(path_key_list[:-1] + [activation_module_original_name]))
            
            try:
                new_riemann_layer = RiemannLinearTransform(
                    in_feats, out_feats, curvature,
                    svd_rank_or_ratio=svd_rank_or_ratio,       # Ensure these are passed
                    fft_compression_ratio=fft_compression_ratio, # Ensure these are passed
                    bias=(initial_bias is not None),
                    initial_weight_data=initial_weight,
                    initial_bias_data=initial_bias,
                    activation_type=activation_to_fuse_type_str,
                    module_name_for_debug=full_module_name_str 
                )
                params_in_new_riemann_layer = sum(p.numel() for p in new_riemann_layer.parameters())
                setattr(model, name, new_riemann_layer)
                _stats_accumulator["replaced_count"] += 1
                _stats_accumulator["current_total_params_after_conversion"] -= params_in_original_target_layer
                fused_activation_display_name = ""
                if activation_to_fuse_type_str and activation_module_original_name:
                    setattr(model, activation_module_original_name, nn.Identity())
                    _stats_accumulator["current_total_params_after_conversion"] -= params_in_fused_activation
                    idx += 1 
                    fused_activation_display_name = '+' + activation_to_fuse_type_str
                _stats_accumulator["current_total_params_after_conversion"] += params_in_new_riemann_layer
            except ValueError as ve:
                 if "Initial bias shape mismatch" in str(ve):
                     print(f"  WARNING: SKIPPING Riemann conversion for {full_module_name_str} due to bias error: {ve}. Original layer kept.")
                     _processed_names.add(full_module_name_tuple)
                 else: raise ve 
            idx += 1
        else:
            _processed_names.add(full_module_name_tuple)
            if len(list(child_module.children())) > 0:
                convert_linear_to_riemann(child_module, curvature, svd_rank_or_ratio, fft_compression_ratio, 
                                        _current_path=path_key_list, 
                                        _processed_names=_processed_names, 
                                        _stats_accumulator=_stats_accumulator)
            idx += 1
    
    if is_top_level_call:
        final_params_before = _stats_accumulator["initial_total_params"]
        final_params_after = _stats_accumulator["current_total_params_after_conversion"]
        print("-" * 70)
        if _stats_accumulator["replaced_count"] > 0:
            actual_compression_ratio = final_params_after / final_params_before if final_params_before > 0 else 0
            print(f"✅ 총 {_stats_accumulator['replaced_count']}개 레이어(군) Riemann (SVD+FFT) 변환 완료")
            print(f"📊 전체 모델 파라미터 변화 (SVD+FFT): {final_params_before:,} → {final_params_after:,} (실제 모델 압축률: {actual_compression_ratio:.2%})")
        elif final_params_before > 0:
            print("✅ Riemann (SVD+FFT) 변환 대상 레이어를 찾지 못했거나, 변환이 이루어지지 않았습니다.")
            print(f"📊 전체 모델 파라미터 (변화 없음): {final_params_before:,}")
        else: print("✅ 모델에 파라미터가 없거나, Riemann (SVD+FFT) 변환 대상 레이어가 없습니다.")
        print("=" * 70)
    return model

def demo_riemann_transformation():
    """리만기하 변환 데모"""
    print("🧮 리만기하 선형+활성화 융합 데모")
    print("=" * 70)

    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.relu1 = nn.ReLU() # 명시적 활성화 레이어
            self.fc2 = nn.Linear(256, 128)
            # fc2 다음에는 활성화 없음 (테스트용)
            self.fc3 = nn.Linear(128, 64)
            self.gelu3 = nn.GELU() # 명시적 활성화 레이어
            self.classifier = nn.Linear(64, 10)

        def forward(self, x):
            x = self.relu1(self.fc1(x))
            x = self.fc2(x) # 활성화 없는 레이어
            x = self.gelu3(self.fc3(x))
            return self.classifier(x)

    original_model = SimpleMLP()
    print("📏 원본 모델 구조:")
    print(original_model)
    original_params = sum(p.numel() for p in original_model.parameters() if p.requires_grad)
    print(f"  총 학습 가능 파라미터: {original_params:,}")

    test_input = torch.randn(16, 128)
    with torch.no_grad():
        original_output = original_model(test_input)
    print(f"  원본 출력 형태: {original_output.shape}")

    # SimpleMLP의 새 인스턴스에 변환 적용
    model_to_convert = SimpleMLP()
    riemann_model = convert_linear_to_riemann(
        model_to_convert, 
        curvature=0.0, 
        svd_rank_or_ratio=None, 
        fft_compression_ratio=0.999
    )

    print(f"\n🧮 리만기하 변환 모델 구조 (Curvature=0, Compression=25%, Linear+Act Fused):")
    print(riemann_model)
    riemann_params = sum(p.numel() for p in riemann_model.parameters() if p.requires_grad)
    print(f"  총 학습 가능 파라미터: {riemann_params:,}")
    if original_params > 0:
        print(f"  파라미터 압축률 달성: {riemann_params/original_params:.2%}")

    with torch.no_grad():
        riemann_output = riemann_model(test_input)
    print(f"  리만 모델 출력 형태: {riemann_output.shape}")

    mse_loss = F.mse_loss(original_output, riemann_output)
    cosine_sim = F.cosine_similarity(original_output.flatten(), riemann_output.flatten(), dim=0)
    print(f"\n📊 변환 품질 (초기 가중치 복사 후, 미세조정 없음):")
    print(f"  MSE 손실 (원본 vs 리만): {mse_loss.item():.6f}")
    print(f"  코사인 유사도 (원본 vs 리만): {cosine_sim.item():.4f}")
    print(f"\n🔬 수학적 변환 원리 요약:")
    print(f"  1. 가중치 압축: Weight_FFT -> TopK 계수 저장")
    print(f"  2. 선형+활성화 융합: Linear/Conv1D + Activation -> RiemannLinearTransform(activation_type)")
    print(f"  3. 내부 연산: 입력매핑 -> 압축해제된 가중치로 변환 -> 지정된 활성화 적용 -> (선택적)활성화 스펙트럴변환 -> 출력매핑")

def demo_korean_text_generation():
    """KoGPT2 (Helgason + Riemann) 테스트"""
    print("\n" + "=" * 70)
    print("🇰🇷 KoGPT2 (Helgason + Riemann) 테스트 - SVD 압축 적용")
    print("=" * 70)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "skt/kogpt2-base-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

        print(f"\n원본 {model_name} 로드 중... ({device})")
        kogpt2_model_instance = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        original_params = sum(p.numel() for p in kogpt2_model_instance.parameters() if p.requires_grad)
        print(f"  원본 모델 파라미터: {original_params:,}")
        
        prompt = "옛날 옛날 아주 먼 옛날에,"
        print("\n원본 모델 생성 테스트:")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            original_outputs = kogpt2_model_instance.generate(
                inputs.input_ids, max_length=60, pad_token_id=tokenizer.eos_token_id,
                do_sample=True, top_k=50, top_p=0.95, temperature=0.7
            )
        original_text = tokenizer.decode(original_outputs[0], skip_special_tokens=True)
        print(f"  프롬프트: \"{prompt}\"")
        print(f"  원본 생성: \"{original_text}\"")

        print(f"\n🔥 {model_name} Helgason 선형 퓨전 중...")
        helgason_fuse_sequential_linear_layers(kogpt2_model_instance, verbose=False) 
        # params_after_helgason = sum(p.numel() for p in kogpt2_model_instance.parameters() if p.requires_grad)
        # print(f"  파라미터 (Helgason 퓨전 후): {params_after_helgason:,}") # Usually no change for KoGPT2

        # SVD 압축 테스트 설정
        svd_setting = 0.7  # SVD 랭크를 70%로 압축
        fft_setting = 0.999 # FFT는 거의 무손실로 유지하여 SVD 효과만 관찰

        print(f"\n🧮 {model_name} (Helgason 퓨전된 모델에) Riemann SVD+FFT 변환 중 (SVD ratio: {svd_setting:.3f}, FFT ratio: {fft_setting:.3f})...")
        # 모델을 복제하여 원본 Helgason 퓨전 모델에 영향 없도록 함
        model_to_convert = AutoModelForCausalLM.from_pretrained(model_name).to(device) # Reload a fresh copy
        helgason_fuse_sequential_linear_layers(model_to_convert, verbose=False) # Apply Helgason to the copy

        riemann_kogpt2 = convert_linear_to_riemann(
            model_to_convert, 
            curvature=0.0, 
            svd_rank_or_ratio=svd_setting, 
            fft_compression_ratio=fft_setting 
        )
        
        riemann_params = sum(p.numel() for p in riemann_kogpt2.parameters() if p.requires_grad)
        print(f"  리만 변환 모델 파라미터 (최종): {riemann_params:,}")
        if original_params > 0:
            print(f"  최종 파라미터 압축률 (vs 원본): {riemann_params/original_params:.2%}") 
        
        print("\n리만 변환 모델 생성 테스트 (SVD 압축 적용):")
        inputs_riemann = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            riemann_outputs = riemann_kogpt2.generate(
                inputs_riemann.input_ids, max_length=60, pad_token_id=tokenizer.eos_token_id,
                do_sample=True, top_k=50, top_p=0.95, temperature=0.7
            )
        riemann_text = tokenizer.decode(riemann_outputs[0], skip_special_tokens=True)
        print(f"  프롬프트: \"{prompt}\"")
        print(f"  리만 생성 (SVD {svd_setting:.1f}, FFT {fft_setting:.3f}): \"{riemann_text}\"")

        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, original_text, riemann_text).ratio()
        print(f"\n📊 생성 품질 (간단 비교):")
        print(f"  텍스트 유사도 (SequenceMatcher): {similarity:.3f}")

    except ImportError: print("Transformers library needed.")
    except Exception as e: print(f"KoGPT2 테스트 중 오류: {e}"); import traceback; traceback.print_exc()

if __name__ == "__main__":
    # demo_riemann_transformation() # SimpleMLP 데모는 일단 건너뛰기 가능
    demo_korean_text_generation() 