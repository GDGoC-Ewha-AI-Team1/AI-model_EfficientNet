import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import os

from src.dataset import RailDataset
from src.model import RailDetectionModel, RailLoss
from src.utils import evaluate_performance, evaluate_metrics
# 설정값
BATCH_SIZE = 16
IMG_SIZE = 256
LEARNING_RATE = 3e-4
EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 경로 설정
CSV_PATH = './data/train.csv' 
IMG_DIR = './data/imagesLevelCrossing'

def main():
    # 1. 데이터 로드
    df = pd.read_csv(CSV_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_df.to_csv('temp_train.csv', index=False)
    val_df.to_csv('temp_val.csv', index=False)

    # 2. 데이터셋
    train_dataset = RailDataset('temp_train.csv', IMG_DIR, IMG_SIZE, 'train')
    val_dataset = RailDataset('temp_val.csv', IMG_DIR, IMG_SIZE, 'valid')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 모델
    model = RailDetectionModel(model_name='efficientnet_b0').to(DEVICE)
    criterion = RailLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 모니터링용
    bce_criterion = nn.BCELoss()
    mse_criterion = nn.MSELoss()
    
    # ★ 최고 기록 저장 변수
    best_val_loss = float('inf')

    # 4. 학습 루프
    print("학습 시작...")
    for epoch in range(EPOCHS):
        model.train()
        
        running_total_loss = 0.0
        running_bce = 0.0
        running_mse = 0.0
        
        loop = tqdm(train_loader)
        for images, targets in loop:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # 모니터링용 단순 계산
            with torch.no_grad():
                prob_idx = [0, 5, 10]
                box_idx = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14]
                bce = bce_criterion(outputs[:, prob_idx], targets[:, prob_idx])
                mse = mse_criterion(outputs[:, box_idx], targets[:, box_idx])
            
            running_total_loss += loss.item()
            running_bce += bce.item()
            running_mse += mse.item()
            
            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())
        
        # Train 결과 출력
        avg_train_loss = running_total_loss / len(train_loader)
        print(f"\n[Epoch {epoch+1}] Train Loss: {avg_train_loss:.5f}")
        print(f"BCE: {running_bce/len(train_loader):.4f}, MSE: {running_mse/len(train_loader):.4f})")

        # utils.py에서 return해준 값을 val_loss 변수에 받음
        val_loss = evaluate_performance(model, val_loader, DEVICE)
        
        if val_loss < best_val_loss:
            print(f"** 모델 저장! (기존: {best_val_loss:.5f} -> 갱신: {val_loss:.5f})")
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            print(f"변화 없음 (Best: {best_val_loss:.5f})")
            
        print("-" * 50)

    evaluate_metrics(model, val_loader, DEVICE)

if __name__ == '__main__':
    main()