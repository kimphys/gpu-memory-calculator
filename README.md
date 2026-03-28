# 🚀 LLM GPU Memory & Performance Simulator

LLM 모델의 VRAM 요구량과 추론/학습 성능을 하드웨어 스펙 기반으로 정밀하게 예측하는 웹/데스크톱 시뮬레이터입니다.

## 핵심 기능 (Core Features)

- **VRAM 점유율 분석**: 모델 가중치, KV 캐시(FP8/FP16), 시스템 예약 메모리, 학습 시 필요한 Optimizer/Activation 메모리를 구분하여 실시간 가시화합니다.
- **추론/학습 성능 추정**: GPU 메모리 대역폭(Bandwidth) 및 연산 능력(TFLOPS)을 바탕으로 Throughput(t/s), TTFT(ms), 최대 동시 추론(Concurrency), RPS를 산출합니다.
- **모델 사양 통합**: 최신 LLM 프리셋(Ministral 3, Qwen 2.5, Llama 3.1 등) 외에 Hugging Face URL 직접 파싱 또는 수동 입력을 지원합니다.
- **MoE(Mixture of Experts) 시뮬레이션**: 전체 파라미터(VRAM 탑재용)와 활성 파라미터(성능 계산용)를 분리하여 MoE 특유의 성능을 정확히 모델링합니다.
- **AWS 인프라 매핑**: 시뮬레이션 결과에 적합한 AWS EC2 인스턴스 타입(P5, P4, G6e, G5)을 추천하고 공식 문서(KR) 링크를 제공합니다.
- **AI 최적화 가이드**: OOM(Memory 부족) 발생 시 양자화 적용, 컨텍스트 길이 조정 등 구체적인 해결 방안을 실시간으로 추천합니다.

## 기술 스택 (Tech Stack)

- **Frontend**: Next.js 15 (App Router), TypeScript, Tailwind CSS
- **Desktop**: Electron (Native Wrapper)
- **State**: Zustand

## 시작하기 (Getting Started)

의존성 설치 후 웹 또는 데스크톱 환경에서 실행할 수 있습니다.

```bash
# 의존성 설치
npm install

# 웹 브라우저 실행 (http://localhost:3000)
npm run dev

# 데스크톱 앱 실행 (Electron)
npm run electron:dev
```

## 커스터마이징 (Customization)

폐쇄망 환경에서 모델 프리셋을 추가하거나 수학 공식을 수정하려면 아래 가이드를 참조하세요.
- [CUSTOMIZATION_GUIDE.md](./CUSTOMIZATION_GUIDE.md)

