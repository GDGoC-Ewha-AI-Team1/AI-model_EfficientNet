import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


from src.dataset import RailDataset
from src.model import RailDetectionModel

# === 설정값 (본인 환경에 맞게 수정) ===
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = './best_model.pth'          # 학습된 모델 가중치 파일 경로
TEST_CSV_PATH = './data/test.csv'        # 테스트 데이터 CSV 경로
IMG_DIR = './data/imagesLevelCrossing'   # 이미지 폴더 경로
OUTPUT_CSV = 'psy_submission.csv'            # 결과 저장할 파일명
IMG_SIZE = 256
BATCH_SIZE = 32

def main():
    print(f"Inference 시작 (Device: {DEVICE})")
    
    # 1. 모델 초기화 및 가중치 로드
    # (주의: 학습할 때 썼던 model_name과 동일해야 함)
    model = RailDetectionModel(model_name='efficientnet_b0', pretrained=False)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"모델 로드 성공: {MODEL_PATH}")
    else:
        print(f"오류: 모델 파일이 없습니다! ({MODEL_PATH})")
        print("   -> python train.py 를 먼저 실행해서 모델을 만드세요.")
        return

    model.to(DEVICE)
    model.eval() # 평가 모드로 설정 (Dropout, Batchnorm 등 고정)

    # 2. Test 데이터셋 & 로더 생성
    # mode='test'로 설정하면 정답(target) 대신 더미 값을 반환함
    test_dataset = RailDataset(
        csv_file=TEST_CSV_PATH,
        img_dir=IMG_DIR,
        img_size=IMG_SIZE,
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=2
    )
    
    print(f"📂 테스트 데이터: {len(test_dataset)}개")

    # 3. 예측 루프
    all_predictions = []
    
    print("예측 수행 중...")
    with torch.no_grad(): # 그래디언트 계산 끔 (메모리 절약)
        for images, _ in tqdm(test_loader):
            images = images.to(DEVICE)
            
            # 모델 예측
            outputs = model(images)
            
            # GPU 텐서 -> CPU 넘파이 변환 후 리스트에 저장
            all_predictions.append(outputs.cpu().numpy())

    # 4. 결과 정리 및 저장
    # 리스트에 쪼개진 배치들을 하나의 큰 배열로 합치기
    predictions = np.concatenate(all_predictions, axis=0)
    
    # 제출용 컬럼명 생성 (proba1, x1, dx1, y1, dy1 ... 반복)
    target_cols = []
    for i in range(1, 4): # 차량 1, 2, 3
        target_cols.extend([f'proba{i}', f'x{i}', f'dx{i}', f'y{i}', f'dy{i}'])
    
    # 예측값 데이터프레임 생성
    pred_df = pd.DataFrame(predictions, columns=target_cols)
    
    # 원본 test.csv 읽어오기 (파일명 컬럼 등을 합치기 위해)
    test_origin = pd.read_csv(TEST_CSV_PATH)
    
    # 원본 데이터 + 예측값 합치기
    submission = pd.concat([test_origin, pred_df], axis=1)
    
    # CSV 저장
    submission.to_csv(OUTPUT_CSV, index=False)
    print(f"저장 완료-> {OUTPUT_CSV}")

if __name__ == '__main__':
    main()