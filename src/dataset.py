#for data preprocessing
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset 
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RailDataset(Dataset):
    def __init__(self, csv_file, img_dir, img_size=256, mode='train'):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.mode = mode
        
        self.transform = self.get_transforms()

    def get_transforms(self):
        # 6채널에 대한 Mean/Std (RGB * 2)
        mean = [0.485, 0.456, 0.406] * 2
        std = [0.229, 0.224, 0.225] * 2

        if self.mode == 'train':
            return A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                # A.ColorJitter(...) # 6채널이라 에러나서 제거함
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 이미지 경로 처리 (try-except는 데이터셋 포맷에 따라 유동적)
        try:
            p1 = os.path.join(self.img_dir, row['image-3sec']) 
            p2 = os.path.join(self.img_dir, row['image'])
        except KeyError:
            p1 = os.path.join(self.img_dir, row.iloc[0]) 
            p2 = os.path.join(self.img_dir, row.iloc[1]) 

        # 이미지 읽기
        img1 = cv2.imread(p1)
        if img1 is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {p1}")
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        
        img2 = cv2.imread(p2)
        if img2 is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {p2}")
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # (H, W, 6) 채널 합치기
        concat_img = np.concatenate([img1, img2], axis=-1)

        # 전처리 적용
        if self.transform:
            transformed = self.transform(image=concat_img)
            image_tensor = transformed['image']

        # 테스트 모드면 정답 없이 이미지만 리턴
        if self.mode == 'test':
             return image_tensor, torch.zeros(15)

        # 정답(Target) 추출 (마지막 15개 컬럼)
        targets = row[-15:].values.astype(np.float32)
        return image_tensor, torch.tensor(targets)