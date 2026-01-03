import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from src.dataset import RailDataset
from src.model import RailDetectionModel

# === 설정값 ===
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = './best_model.pth'
TEST_CSV_PATH = './data/test.csv'
IMG_DIR = './data/imagesLevelCrossing'
OUTPUT_CSV = 'psy_submission.csv'
IMG_SIZE = 256
BATCH_SIZE = 32

def main():
    print(f"Inference 시작 (Device: {DEVICE})")
    
    # 1. 모델 초기화 및 가중치 로드
    model = RailDetectionModel(model_name='efficientnet_b0', pretrained=False)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"모델 로드 성공: {MODEL_PATH}")
    else:
        print(f"오류: 모델 파일이 없습니다! ({MODEL_PATH})")
        return

    model.to(DEVICE)
    model.eval()

    # 2. Test 데이터셋 로드
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
    
    print(f"테스트 데이터: {len(test_dataset)}개")

    # 3. 예측 수행
    all_predictions = []
    
    with torch.no_grad():
        for images, _ in tqdm(test_loader):
            images = images.to(DEVICE)
            outputs = model(images)
            all_predictions.append(outputs.cpu().numpy())

    # 4. 결과 정리 (제출 양식 맞춤)
    predictions = np.concatenate(all_predictions, axis=0)
    
    # 제출 파일 컬럼명 생성 (sample_submission.csv 기준)
    target_cols = []
    for i in range(1, 4):
        target_cols.extend([f'probaObstacle{i}', f'x{i}', f'dx{i}', f'y{i}', f'dy{i}'])
    
    # 예측값 데이터프레임 생성
    submission = pd.DataFrame(predictions, columns=target_cols)
    
    # 원본 test.csv에서 ID 컬럼만 가져와서 맨 앞에 붙이기
    test_origin = pd.read_csv(TEST_CSV_PATH)
    submission.insert(0, 'ID', test_origin['ID'])
    
    # 저장
    submission.to_csv(OUTPUT_CSV, index=False)
    print(f"저장 완료 -> {OUTPUT_CSV}")

if __name__ == '__main__':
    main()