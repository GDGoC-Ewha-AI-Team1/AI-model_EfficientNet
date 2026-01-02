import torch
import torch.nn as nn
import timm

class RailDetectionModel(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=True):
        super(RailDetectionModel, self).__init__()
        
        # 1. 백본(Backbone) 모델 가져오기 (EfficientNet)
        # in_chans=6 : 이미지를 2장 겹쳐서 넣을 거라 채널이 6개
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            in_chans=6, 
            num_classes=0  # 분류기(Head) 제거하고 특징만 뽑음
        )
        
        # 백본 모델이 내뱉는 특징 개수 자동 계산
        in_features = self.backbone.num_features
        
        # 2. 헤드(Head) 정의 (15개 값 예측)
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 15), # 최종 출력 15개
            nn.Sigmoid()        # 0~1 사이 값으로 변환
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output

class RailLoss(nn.Module):
    def __init__(self):
        super(RailLoss, self).__init__()
        self.bce = nn.BCELoss() # 확률 오차
        self.mse = nn.MSELoss() # 좌표 오차

    def forward(self, preds, targets):
        # 1. 존재 확률 (인덱스 0, 5, 10)
        prob_idx = [0, 5, 10]
        loss_probs = self.bce(preds[:, prob_idx], targets[:, prob_idx])
        
        # 2. 좌표 (나머지 인덱스)
        coord_idx = [1,2,3,4, 6,7,8,9, 11,12,13,14]
        loss_coords = self.mse(preds[:, coord_idx], targets[:, coord_idx])
        
        return loss_probs + loss_coords