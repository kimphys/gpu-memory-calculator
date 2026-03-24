# LLM GPU 메모리 & 성능 시뮬레이터 프로젝트 기획서

## 1. 프로젝트 개요 (Overview)

### 1-1. 배경 및 목적
- **현상**: 영업, 기획, PM, AI 컨설턴트 등 비개발 직군에서 빈번하게 "특정 LLM(Llama 3, Mistral, Qwen 등)이 특정 GPU(A100, H100 등) N장에서 학습/추론이 가능한가요?", "속도는 어느 정도 나오나요?"와 같은 질문이 기술팀/엔지니어링팀으로 쏟아지고 있습니다.
- **문제점**: 매번 엔지니어가 수작업으로 계산하거나 파라미터를 찾아야 해서 리소스가 낭비되고, 영업 현장에서는 고객의 질문에 즉각적인 답변을 실시간으로 주기 어렵습니다.
- **해결 방안**: 파라미터(모델 종류, 정밀도)와 하드웨어 환경(GPU 종류, 수량)을 선택하면, 직관적으로 OOM(Out of Memory) 발생 여부와 예상 성능(TPS)을 시뮬레이션 해주는 전용 웹 서비스를 구축합니다.

### 1-2. 핵심 가치 (Value Proposition)
- **Time-to-Market 단축**: 영업 및 기획 과정에서의 소통 지연 제거, 실시간 답변을 통한 고객 신뢰도 확보.
- **엔지니어 리소스 절약**: 반복적인 단순 계산성 질문을 시스템이 흡수, 본연의 R&D 업무에 집중.
- **비용 최적화 가이드**: 필요 이상으로 높은 사양의 GPU를 제안하는 오버스펙(Over-spec)을 방지하고 최적의 인프라 구성 추천.

---

## 2. 타겟 유저 및 User Story

- **영업 담당자**: "고객사가 온프레미스로 A100 80GB 2장을 가지고 있는데, 여기에 Llama-3 70B 모델을 올릴 수 있는지 바로 확인해서 견적서를 작성하고 싶다."
- **기획자 / PM**: "새로운 서비스에 Qwen 72B 모델을 도입하려고 하는데, 양자화(INT8/INT4)를 적용하면 H200 1장으로도 충분한지 사전에 검토하고 싶다."
- **AI 엔지니어**: "복잡한 수식을 직접 계산하기보다 빠르고 간편하게 Batch Size에 따른 KV Cache 메모리 증가량을 확인하고 싶다."

---

## 3. 핵심 기능 요구사항 (Core Features)

### 3-1. 시뮬레이션 입력 단 (Input)
- **1) 모델 정의 (Model Definition) - 3가지 옵션 제공**
  - **옵션 A (프리셋 선택)**: Llama (v2, v3), Mistral, Qwen, Gemma, GPT-OSS 등 주요 모델 프리셋 제공 (파라미터 및 기본 설정 내장).
  - **옵션 B (Hugging Face URL 입력)**: 사용자가 HuggingFace Model 허브 URL (예: `meta-llama/Meta-Llama-3-8B`)을 입력하면, 시스템이 자동으로 파싱하여 모델 파라미터(Hidden size, Num layers 등)를 추출해 시뮬레이션.
  - **옵션 C (직접 입력)**: 총 파라미터 수(e.g., 70B, 8B) 등 시뮬레이션에 필요한 필수 정보를 사용자가 수동 작성.
- **2) 정밀도 및 양자화 설정 (Precision & Quantization)**
  - 모델 가중치에 대한 정밀도: FP16 / BF16 (기본), INT8, INT4(AWQ, GPTQ 등).
- **2) GPU 프리셋 선택 (Hardware Selection)**
  - 지원 라인업: NVIDIA A100 (40GB/80GB), H100 (80GB), H200 (141GB), L40S (48GB), A30 (24GB), RTX 4090/3090 (24GB) 등 주요 서버/워크스테이션용 GPU.
  - **GPU 개수**: 1장 ~ 8장 이상 (멀티 GPU).
- **3) 작업(Task) 및 하이퍼파라미터 입력**
  - **Task 선택**: Inference (추론) vs Training / Fine-tuning (학습).
  - Inference 파라미터: Context Length(입력 토큰 길이), Batch Size, Max Generated Tokens.
  - Training 파라미터: 학습 방식 (Full Fine-tuning, LoRA, QLoRA 등), Batch Size, Optimizer 종류 (AdamW 등).

### 3-2. 시뮬레이션 결과 단 (Output / Analytics)
- **1) VRAM 메모리 분석 (Memory Profiling)**
  - Model Weights Memory (모델 가중치 로드에 필요한 용량).
  - KV Cache Memory (추론 시 Context Length와 Batch Size에 비례하는 메모리).
  - Activations / Optimizer States (학습 시 필요한 추가 메모리).
  - **직관적 결과 노출**: 
    - 🟢 "충분합니다" (Safe)
    - 🟡 "아슬아슬합니다 (OOM 주의)" (Warning)
    - 🔴 "OOM 발생! GPU가 부족합니다" (Fail)
- **2) 성능/속도 추정 (Performance Estimation)**
  - Memory Bandwidth 기반 **초당 토큰 생성 속도(TPS: Tokens Per Second)** 추정치 제공.
  - **멀티 GPU 병목 안내**: Tensor Parallelism (TP) 적용 시 통신 오버헤드 반영 및 유의사항 메세지 노출.
- **3) 최적의 대안 추천 (Actionable Insights)**
  - (OOM 발생 시) "INT8 양자화를 적용하면 현재 GPU 구성으로도 구동이 가능합니다."
  - (vRAM 여유 시) "현재 구성에서는 동시에 Max Batch Size N까지 처리 가능합니다."

### 3-3. 부가 기능
- **Share (공유 기능)**: 시뮬레이션 결과를 이미지나 고유 링크로 캡처하여 바로 슬랙/이메일로 전달.
- **Glossary (용어 사전 툴팁)**: KV Cache, INT4, LoRA 등 전문 용어에 마우스를 올리면 쉬운 설명 팝업 노출.

---

## 4. 화면 및 UX/UI 설계 방향성

1. **대시보드 형태의 2-Column/Split Layout**
   - **좌측 (설정 패널)**: 드롭다운 메뉴와 슬라이더를 사용하여 모델/GPU/Context Length를 조작. 사용자가 값을 바꿀 때마다 새로고침 없이(Interactive) 우측 결과가 실시간 업데이트.
   - **우측 (결과 패널)**: 큰 게이지 차트(Gauge Chart)나 누적 막대 차트(Stacked Bar)를 활용해 "총 VRAM 확보량 vs 필요량"을 시각화.
2. **시각적 경고 시스템**
   - VRAM 사용량이 100%를 초과하는 순간 그래프가 빨간색으로 변하고 강력한 엑스(X) 표시 노출.
3. **토글 버튼 활용**
   - 상단에 "추론(Inference) 모드" ↔ "학습(Training) 모드"를 크게 분리하여 불필요한 입력값 숨김 처리.

---

## 5. 시스템 아키텍처 및 기술 스택

- **Frontend**: `Next.js` (React), `Tailwind CSS`, `Framer Motion` (부드러운 시각적 전환/애니메이션), 시각화(Chart.js, Recharts 등).
- **Backend**: (초기 MVP 단계에서는 불필요). 로직이 계산식으로 명확히 떨어지므로 100% Client-side(브라우저)에서 계산 처리하여 서버 비용 0원 유지 및 빠른 응답성 확보 가능.
- **배포망 (Deployment)**: Vercel 또는 Netlify (무료 글로벌 CDN 캐싱으로 초고속 로딩).

---

## 6. 프로젝트 마일스톤 (Phases)

| 페이즈 | 주요 목표 | 포함 기능 | 예상 마감 기한 |
|---|---|---|---|
| **Phase 1 (MVP)** | 주요 영업/기획 질문 80% 방어 | 인기 모델 5~10선, 주요 GPU 프리셋, **Inference(추론) 메모리 계산** 기능 중심. | 1~2 주 |
| **Phase 2 (고도화)** | 엣지 케이스 및 학습 검증 | **Training(학습) VRAM 계산** (LoRA/QLoRA 지원), Custom 모델 입력 기능. 예상 Inference TPS 수치 산출 모델 고도화. | MVP 오픈 후 2주 |
| **Phase 3 (확장)** | 추천 자동화 및 B2B 툴화 | OOM 시 **자동 대안 추천 알고리즘** 구현, 결과 리포트 PDF 다운로드 및 슬랙 공유 기능. UI/UX 디자인 폴리싱. | Phase 2 이후 2주 |

---

## 7. 주요 수식 (참고자료)

시뮬레이터 로직 구현을 위해 사용될 핵심 수식입니다 (단위: Bytes).

1. **가중치 메모리 (Model Weights)**
   - `Parameters * Bytes per parameter` (FP16/BF16: 2 bytes, INT8: 1 byte, INT4: 0.5 bytes)
2. **KV Cache (Inference)** (Transformer 기준)
   - `2 (K, V) * Batch_Size * Context_Length * hidden_size * num_layers * Bytes_per_parameter`
   - (GQA/MQA 적용 모델의 경우 head 비율에 따라 KV Cache 메모리가 약 절반~수 분의 일로 대폭 감소)
3. **통신 오버헤드 (Multi-GPU)**
   - TP를 사용할 때 모델 가중치는 N분의 1로 나뉘지만 로컬 Activation 등의 변수가 발생. 관련 경험 법칙(Rule of thumb) 적용하여 여유공간 10~20% 할당 필수 조언.
