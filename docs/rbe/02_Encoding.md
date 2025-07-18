# 2. 인코딩: 파라미터 → `Packed64` 시드 (`src/encoding.rs`)

이 문서에서는 여러 개의 생성 파라미터(반지름, 각도, 기저 함수 ID 등)를 단 하나의 64비트 정수(`u64`)로 압축하는 인코딩 과정을 설명합니다.

---

### `Packed64::new()`

```rust
impl Packed64 {
    pub fn new(
        r: f32, theta: f32, basis_id: u8, d_theta: u8,
        d_r: bool, rot_code: u8, log2_c: i8, reserved: u8
    ) -> Self {
        // ... (구현)
    }
}
```

-   **역할**: 행렬 생성을 위한 모든 파라미터를 입력받아, 비트 연산을 통해 `Packed64` 시드 하나를 생성합니다.
-   **핵심 과정**:
    1.  **정규화 (Normalization)**: `f32` 타입의 `r`과 `theta` 값을 특정 범위(r: `[0, 1)`, theta: `[0, 2π)`)로 맞춥니다.
    2.  **양자화 (Quantization)**: 정규화된 실수 값들을 각 필드에 할당된 비트 수에 맞는 정수로 변환합니다. 예를 들어, `r`은 20비트 정수로, `theta`는 24비트 정수로 변환됩니다.
    3.  **비트 패킹 (Bit Packing)**: 양자화된 정수들을 비트 단위 `SHIFT` (`<<`) 및 `OR` (`|`) 연산을 사용하여 `u64`의 정해진 위치에 차례대로 삽입합니다.

### 64비트 레이아웃

`Packed64` 시드의 64비트는 다음과 같이 정교하게 분할되어 사용됩니다.

| Bit 구간 | 길이 | 필드           | 설명                                                             |
| :------- | :--- | :------------- | :--------------------------------------------------------------- |
| `63–44`  | 20   | **`r`**        | 중심 반지름. 패턴의 전체적인 크기와 진폭을 결정.                 |
| `43–20`  | 24   | **`theta`**    | 중심 각도. 패턴의 전체적인 위상(phase)을 결정.                   |
| `19–16`  | 4    | **`basis_id`** | 사용할 기저 함수(`BasisFunction`)를 선택 (0-15).                 |
| `15–14`  | 2    | **`d_theta`**  | 각도(`θ`)에 대한 미분 차수 (0-3차). 주기 함수의 형태를 바꿈.     |
| `13`     | 1    | **`d_r`**      | 반지름(`r`)에 대한 미분 차수 (0-1차). 쌍곡선 함수의 형태를 바꿈. |
| `12–9`   | 4    | **`rot_code`** | 전체 패턴을 회전시킬 각도를 선택 (0-15).                         |
| `8–6`    | 3    | **`log2_c`**   | 곡률(`c`)을 `log₂` 스케일로 표현. -4부터 +3까지의 값을 가짐.     |
| `5–0`    | 6    | **`reserved`** | 예약된 공간. 향후 추가 기능을 위해 비워 둠.                      |

이러한 비트 패킹을 통해, 단 8바이트 안에 행렬의 복잡한 패턴을 생성하기 위한 모든 정보가 효율적으로 담기게 됩니다. 