from dataclasses import dataclass
@dataclass
class AdvancedConfig:
    enable_dynamic_curvature: bool = False
    base_curvature: float = 1.0 