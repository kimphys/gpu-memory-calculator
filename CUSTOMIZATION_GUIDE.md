# 🛠️ LLM GPU Memory Simulator - 완벽 커스터마이징 가이드 (오프라인 망사용자용)

이 문서는 외부 인터넷 연결이나 AI 어시스턴트의 도움(프롬프팅) 없이도 개발자/기획자 본인이 직접 **시뮬레이터의 모델, 하드웨어 스펙, 수학 공식을 자유자재로 뜯어고칠 수 있도록** 작성된 초정밀 가이드입니다.

---

## 1. 모델(Models) 프리셋 수정 및 추가하기

UI 화면 좌측 "Model Definition -> Preset" 드롭다운에 표시되는 모델 목록은 단일 데이터 파일에 하드코딩되어 있습니다. 인터넷이 안되는 폐쇄망 환경이라면 오직 이 파일을 수정하여 신규 모델을 추가해야 합니다.

- **편집해야 할 파일**: `src/data/presets.ts`

### 1-1. 일반 Dense 모델 추가 문법
파일 내부의 `MODEL_PRESETS` 객체 안에 새로운 객체를 넣어주면 끝납니다.

```typescript
export const MODEL_PRESETS: Record<string, ModelConfig & { name: string }> = {
  // 예시: 새로운 일반 Dense 모델 
  'llama-4-100b': {
    name: 'Llama 4 (100B)',         // (필수) 화면 메뉴에 보일 이름 지정
    parametersInB: 100,             // (필수) 총 파라미터 수 (Billion) -> VRAM 계산의 핵심 체급 데이터
    hiddenSize: 8192,               // (필수) Hidden Dimension 텐서 폭. (설계서나 config.json 참조)
    numLayers: 80,                  // (필수) 전체 Transformer 레이어 수
    numAttentionHeads: 64,          // (필수) Query(Q)를 나누는 Attention Head 수 
    numKeyValueHeads: 8,            // (필수) K,V 캐시를 나누는 수. MQA/GQA 튜닝이 안된 구형 모델은 numAttentionHeads와 똑같이 적으세요.
  }
};
```

### 1-2. 특수 MoE (Mixture of Experts) 모델 추가 문법
가중치 전체 덩치(Total VRAM)와 동작 속도 체급(Active Params)이 완전히 따로 노는 **MoE 계열** 모델은 **단 한 줄(`activeParametersInB`)**을 추가해야 치명적인 속도 계산 오류가 방지됩니다.

```typescript
// 예시: Mixtral 같은 MoE 아키텍처 모델
  'mixtral-8x7b': {
    name: 'Mixtral 8x7B (MoE)',
    parametersInB: 47,              // (필수) VRAM에 통째로 복사되어야 할 총 덩치 (47B)
    activeParametersInB: 13,        // (MoE 핵심!) 토큰 1개 생성 시 실제로 활성화되어 램 대역폭을 긁어내는 파라미터 수 (13B). 이 값을 꼭 넣어야 예상 속도(TPS)가 MoE 특성에 맞게 매우 빠르게 제대로 나옵니다.
    hiddenSize: 4096,
    numLayers: 32,
    numAttentionHeads: 32,
    numKeyValueHeads: 8,
  }
```

---

## 2. 하드웨어(GPU) 프리셋 변경 및 인프라 기기 추가하기

NVIDIA의 새로운 아키텍처나 타사의 강력한 AI 가속기를 메뉴에 추가하여 시뮬레이션을 돌리고 싶을 때 수정하는 영역입니다.

- **편집해야 할 파일**: `src/data/presets.ts`

### 2-1. GPU 데이터 추가 문법
코드 내 `GPUS` 배열에 새로운 항목을 쉼표(,)로 구분하여 블록을 꽂아 넣으세요.

```typescript
export const GPUS = [
  { 
    id: 'h200-141', 
    name: 'NVIDIA H200 (141GB)', 
    vramGb: 141,                    // (필수) 해당 GPU 1장이 보유한 물리적 총 VRAM (기초 단위 GB)
    bandwidthGbps: 4800             // (매우 중요) 메모리 대역폭(Memory Bandwidth, 초당 전송량 GB/s). 초당 몇 토큰을 밀어낼 수 있는지 한계 속도를 계산하는 심장부입니다. 위키나 제조사 스펙시트를 찾아서 기입하세요.
  },
];
```

---

## 3. 시뮬레이터 코어 수학 공식 (로직 엔진) 뜯어고치기

시뮬레이터의 VRAM 계산과 속도(TPS) 추정을 담당하는 수학 공식들은 컴포넌트나 잡다한 UI 데이터와 완벽하게 격리되어, 단일 순수 함수 파일(`.ts`)에 담겨 있습니다.
이론적 한계치가 아닌 **우리 회사 클러스터만의 vLLM 실측 데이터 보정치** 등을 넣고 싶다면 주저 말고 이 함수 로직들을 수정하십시오.

- **편집해야 할 파일**: `src/utils/calculator.ts`

### 3-1. 가중치 VRAM 공식: `calculateWeightsMemory()`
```typescript
function calculateWeightsMemory(parametersInB: number, quantizationBits: number) {
  // 현재: 파라미터 개수 × (정밀도 Bit수 ÷ 8) 
  // 예: 8B × (16bit / 8) = 16GB
  // ★수정 팁★: 만약 INT4 양자화(AWQ, GPTQ) 시, 메타데이터와 패킹 과정에서 발생하는 VRAM 낭비/오버헤드 보정치(예: 1.05배 부풀리기)를 로직에 추가하고 싶다면 이 함수의 return 값에 수식을 박아넣으세요.
}
```

### 3-2. KV Cache 수식: `calculateKVCache()`
```typescript
function calculateKVCache(config, params) {
  // 현재: 2 × Batch × Context Length × KV Heads × Head Dimension × Layers × 2(bytes)
  // ★수정 팁★: 이 계산식은 vLLM PagedAttention 환경(낭비율 0%)을 기본 전제로 한 최고 이상치입니다. 만약 파편화(Fragmentation) 페널티가 생기는 일반 PyTorch/HF TGI 보수적 환경을 구상한다면, return 값에 1.25배 가량의 페널티 상수를 곱해버리면 현실적인 OOM 상황을 더 빨리 경고할 수 있습니다.
}
```

### 3-3. 추론 생성 속도 산출 로직: `estimateTPS()`
```typescript
function estimateTPS() {
  // 현재: (GPU대역폭 × 통신효율) ÷ (Active 가중치 데이터 사이즈)
  // ★수정 팁★: 멀티 GPU 시, 현재는 Tensor Parallel 효율을 단순하게 뭉뚱그려 "80% (0.8)"로 통일해 깎고 있습니다. 
  // 그러나 만약 NVLink 4.0 이냐 구형 PCIe 4.0 이냐에 따라 섬세하게 통신 효율을 나누어 계산하고 싶다면, 이 함수의 상단 인자(Args) 쪽으로 Topology 값을 받게끔 파라미터를 하나 뚫어주시고 If-else 처리를 태워주면 완벽합니다.
}
```

---

## 4. UI 및 테마 색상(Theme) 수정 가이드

- **색상 팔레트 수정 장소**: `src/app/globals.css`
  파일 상단 `@theme` 블록 안에 선언된 `--color-primary-*` (현재 청록색) 및 `--color-accent-*` (현재 보라색) 컬러의 HEX 코드를 맘대로 변경하시면 UI 전체 테마가 물듭니다. OOM 시 뿜어내는 붉은 경고가 마음에 안 든다면 `--color-error` 의 HEX 값을 조절하세요.
- **차트 비율 게이지 바 수학 로직**: `src/components/ResultsDashboard.tsx` 파일 내 하단부.
  화면 오른쪽의 Stacked Bar(비율 차트) HTML 코드를 보시면 `width: \${weightsPct}%` 방식으로 CSS 자체 인라인을 통해 퍼센트 렌더링이 이루어지고 있습니다. 차트 최대 한계 스케일을 바꾸려면 `maxScale` 변수 선언부를 개조하면 됩니다.

---

## (참고) 인터넷 연결이 끊겼을 때의 Hugging Face 파서
- **해당 파일**: `src/utils/huggingface.ts`
- 폐쇄망(오프라인)에서는 이 파일의 API `fetch()` 호출이 실패하며 조용히 NULL을 반환하도록 예외처리가 안전하게 방어막 쳐져 있습니다. 굳이 건드리지 않아도 시뮬레이터 프로그램 전체가 다운되지는 않으니, 오프라인 망에서는 쿨하게 `Custom(직접 입력)` 탭이나 **1번 항목의 프리셋 조작** 기능만 100% 신뢰하여 사용하십시오.
