
# 🚂 Rail Crossing Vehicle Detection (철도 건널목 차량 탐지)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

이 프로젝트는 철도 건널목(Level Crossing) CCTV 영상 데이터를 활용하여 **선로 위에 정차된 차량을 실시간으로 탐지**하는 AI 모델입니다.  
연속된 두 프레임의 이미지를 분석하여 **차량의 위치(Bounding Box)** 와 **존재 확률**을 예측합니다.

---

## 📌 Project Overview

- **Goal**: 철도 건널목 사고 방지를 위한 정지 차량 탐지
- **Model**: EfficientNet-B0 (Pretrained)
- **Input**: 연속된 2장의 이미지  
  (6-channel Input: t-1 frame + t-current frame)
- **Output**: 차량 3대에 대한 존재 확률 및 Bounding Box 좌표  
  (총 15개 값 예측)

---

## 📂 Project Structure

아래와 같은 파일 구조를 만든 후 코드를 실행해 주세요.

    Rail-Detection/
    │
    ├── data/
    │   ├── imagesLevelCrossing/
    │   ├── train.csv
    │   └── test.csv
    │
    ├── src/
    │   ├── __init__.py
    │   ├── dataset.py
    │   ├── model.py
    │   └── utils.py
    │
    ├── train.py
    ├── inference.py
    ├── requirements.txt
    └── README.md

---

## 🛠️ Installation & Setup

이 프로젝트는 **VS Code 및 Python 3.8 이상** 환경을 권장합니다.

### 1. 가상환경 생성 및 활성화

    python -m venv venv

Windows:

    .\venv\Scripts\activate

Mac / Linux:

    source venv/bin/activate

### 2. 라이브러리 설치

    pip install -r requirements.txt

requirements.txt 파일이 없는 경우:

    pip install numpy pandas matplotlib opencv-python albumentations torch torchvision timm tqdm scikit-learn

---

## 📊 Data Preparation

⚠️ 저작권 및 용량 문제로 이미지 데이터는 GitHub에 포함되어 있지 않습니다.

- Dataset: Vehicle Stopped on a Level Crossing (Kaggle)
- 다운로드 후 data/ 폴더에 배치하세요.

    Project/
    └── data/
        ├── imagesLevelCrossing/
        ├── train.csv
        └── test.csv

---

## 🚀 Usage

### 1. Model Training

    python train.py

- 학습 완료 후 best_model.pth 생성

### 2. Inference

    python inference.py

- submission.csv 파일 생성

---

## 🧠 Model Architecture

### Backbone

- EfficientNet-B0

### Input Modification

- RGB 3채널 대신
- 시간차가 있는 2장의 이미지를 채널 방향으로 결합하여 6채널 입력 사용

### Head Structure

    Linear(1280 → 512) → ReLU → Dropout(0.3)
    Linear(512 → 128) → ReLU
    Linear(128 → 15) → Sigmoid

### Output (15 values)

- 차량 3대 각각:
  - Probability
  - Center X
  - Center Y
  - Width
  - Height


## 📈 Performance Evaluation

- MSE (Mean Squared Error): Bounding Box 좌표 정확도
- F1-Score: 차량 존재 여부 분류 성능

## 👤 Author
Name: [박선영]

Contact: [psuny17@ewha.ac.kr]

GitHub: [sunyp17]

---
