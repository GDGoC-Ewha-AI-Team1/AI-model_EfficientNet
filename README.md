# 🚂 Rail Crossing Vehicle Detection (철도 건널목 차량 탐지)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

이 프로젝트는 철도 건널목(Level Crossing) CCTV 영상 데이터를 활용하여, 선로 위에 정차된 차량을 실시간으로 탐지하는 AI 모델입니다.
연속된 두 프레임의 이미지를 분석하여 차량의 위치(Bounding Box)와 존재 확률을 예측합니다.

## 📌 Project Overview
- **Goal**: 철도 건널목 사고 방지를 위한 정지 차량 탐지
- **Model**: EfficientNet-B0 (Pretrained)
- **Input**: 연속된 2장의 이미지 (6-channel Input: $t_{-1}$ frame + $t_{current}$ frame)
- **Output**: 차량 3대에 대한 존재 확률 및 Bounding Box 좌표 (총 15개 값 예측)

## 📂 Project Structure
```bash
Rail-Detection/
│
├── data/                  # 데이터 폴더 (.gitignore 처리됨)
│   ├── imagesLevelCrossing/   # 원본 이미지 폴더 (*.jpg)
│   ├── train.csv              # 학습 데이터 라벨
│   └── test.csv               # 테스트 데이터
│
├── src/                   # 소스 코드 모듈
│   ├── __init__.py
│   ├── dataset.py         # 데이터셋 로더 (RailDataset)
│   ├── model.py           # 모델 정의 (EfficientNet Based)
│   └── utils.py           # 평가 및 시각화 도구
│
├── train.py               # 모델 학습 실행 파일
├── inference.py           # 예측 및 결과 제출 파일
├── requirements.txt       # 필요 라이브러리 목록
└── README.md              # 프로젝트 설명서