# 🚀 LLM GPU Memory & Performance Simulator

![Next.js](https://img.shields.io/badge/Next.js-15-black?logo=next.js&logoColor=white) 
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?logo=tailwind-css&logoColor=white) 
![Zustand](https://img.shields.io/badge/Zustand-React_State-blue)

**LLM GPU Memory & Performance Simulator**는 AI 엔지니어뿐만 아니라 기획자, 영업 담당자, 프로젝트 매니저(PM) 등 IT 직군 누구나 **"이 LLM 모델을 우리 GPU 인프라에 올리면 돌아갈까? 속도는 초당 몇 토큰이나 나올까?"**라는 질문에 즉각적으로 대답할 수 있게 해주는 100% Client-side 웹 대시보드입니다.

---

## ✨ 핵심 기능 (Key Features)

- 🧠 **원클릭 스마트 프리셋 (Presets)**: `Llama 3`, `Qwen 2`, `Gemma 2` 등 최신 주력 오픈소스 모델과 A100, H100, RTX4090 등 주요 NVIDIA GPU 라인업 스펙이 사전 내장되어 있습니다.
- ⚡ **MoE(Mixture of Experts) 완벽 지원**: `Mixtral 8x7B` 같은 MoE 모델의 **총 파라미터(Total VRAM 탑재용)**와 **활성 파라미터(Active 속도 측정용)**를 분리 계산하여, MoE 아키텍처 특유의 압도적인 생성 속도(TPS)를 완벽하게 예측합니다.
- 🔗 **Hugging Face 자동 파싱 로직**: 브라우저에서 허깅페이스 모델 주소(예: `meta-llama/Meta-Llama-3-8B`)를 입력하면 별도의 백엔드 없이 직접 `config.json`을 파싱해 파라미터와 캐시 사이즈를 추출합니다.
- 📊 **실시간 게이지 시각화**: 현재 선택된 하드웨어 스펙에 맞춰 VRAM 소모량(Weights, KV Cache, Training Overhead)을 즉각적인 동적 게이지바로 보여주고, **OOM(Out of Memory)** 상황 시 직관적인 붉은색 경고등을 띄웁니다.
- 💡 **AI 최적화 추천 (Actionable Insights)**: 만약 VRAM이 넘친다면, "INT4(4-bit) 양자화를 적용해 보세요", "컨텍스트 길이를 줄이세요", "Full Finetuning 대신 LoRA를 써보세요" 등의 가이드를 화면 하단에 즉시 실시간으로 렌더링합니다.

---

## 🎯 왜 만들었나요? (Why We Built This?)

새로운 대규모 언어 모델(LLM)이 쏟아질 때마다, B2B 영업 현장이나 사내 서비스 기획 회의에서 가장 먼저 던져지는 질문은 다음과 같습니다.
> *"저희 회사가 가진 A100 80GB 2장에 저 모델 올라가나요?"*  
> *"동시에 몇 명이나 처리할 수 있죠? 속도는 잘 나오나요?"*

이 시뮬레이터는 저러한 단순 견적 질문 하나에 대답하기 위해 엔지니어들이 엑셀표를 열고 일일이 수동으로 계산(파라미터 비트 곱하기, KV캐시 구하기 등)을 하거나, 실제로 무거운 파이토치(PyTorch)를 돌려보다 터뜨려(OOM) 보는 삽질을 완전히 없애기 위해 탄생했습니다.

vLLM(PagedAttention) 및 연속 배치 처리를 사용한다는 가정하의 **가장 이상적이고 최적화된 하드웨어 한계 스피드**를 클릭 몇 번만으로 직관적으로 검증해 보세요!

---

## 🚀 빠른 시작 가이드 (Getting Started)

본 프로젝트는 Next.js (App Router) 기반으로 서버 유지비(Backend-free) 없이 100% 브라우저 연산으로 가볍게 동작합니다.

```bash
# 1. 패키지 설치
npm install

# 2. 로컬 개발 서버 구동
npm run dev
```

터미널 입력 후 브라우저에서 `http://localhost:3000` 에 접속하여 예쁜 글래스모피즘(Glassmorphism) 기반의 시뮬레이터를 바로 구동하실 수 있습니다!

---

## 🛠️ 유지보수 및 커스터마이징 가이드

인터넷이 터지지 않는 꽉 막힌 폐쇄망(오프라인) 환경에서 외부 통신 없이 **우리 회사만의 프라이빗 모델**을 추가하거나, **차세대 신형 하드웨어(예: RTX 5090)**를 메뉴에 넣고 싶으신가요? 

개발자와 운영 기획자를 위한 가장 직관적인 수정 설명 매뉴얼을 별도의 파일로 빼놓았습니다.
👉 **[CUSTOMIZATION_GUIDE.md](./CUSTOMIZATION_GUIDE.md)** 파일을 꼭 읽어주세요! (수학적 통신 지연 효율까지 수정할 수 있습니다.)
