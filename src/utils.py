import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score

def evaluate_performance(model, val_loader, device):
    model.eval()

    total_loss = 0
    total_bce = 0 # 존재 여부(확률) 오차
    total_mse = 0 # 위치(좌표) 오차

    criterion_bce = torch.nn.BCELoss()
    criterion_mse = torch.nn.MSELoss()

    print("성능 평가 중...")

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            preds = model(images)

            # 1. 인덱스 분리
            prob_idx = [0, 5, 10]
            coord_idx = [1,2,3,4, 6,7,8,9, 11,12,13,14]

            # 2. 개별 Loss 계산
            loss_probs = criterion_bce(preds[:, prob_idx], targets[:, prob_idx])
            loss_coords = criterion_mse(preds[:, coord_idx], targets[:, coord_idx])

            total_bce += loss_probs.item()
            total_mse += loss_coords.item()
            total_loss += (loss_probs + loss_coords).item()

    # 평균 계산
    avg_bce = total_bce / len(val_loader)
    avg_mse = total_mse / len(val_loader)
    avg_total = total_loss / len(val_loader)

    print("\n" + "="*40)
    print(f"Total Loss : {avg_total:.5f}")
    print(f"----------------------------------------")
    print(f"- 확률 오차 (BCE) : {avg_bce:.5f}")
    print(f"- 좌표 오차 (MSE) : {avg_mse:.5f}")
    print("="*40)

    return avg_total



def evaluate_metrics(model, val_loader, device):
    model.eval()

    all_pred_probs = []
    all_target_probs = []
    all_pred_coords = []
    all_target_coords = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)

            preds = outputs.cpu().numpy()
            targets = targets.numpy()

            # 확률 분리
            prob_indices = [0, 5, 10]
            all_pred_probs.extend(preds[:, prob_indices].flatten())
            all_target_probs.extend(targets[:, prob_indices].flatten())

            # 좌표 분리
            coord_indices = [1,2,3,4, 6,7,8,9, 11,12,13,14]
            all_pred_coords.extend(preds[:, coord_indices].flatten())
            all_target_coords.extend(targets[:, coord_indices].flatten())

    # 점수 계산
    mse_score = mean_squared_error(all_target_coords, all_pred_coords)

    binary_preds = [1 if p > 0.5 else 0 for p in all_pred_probs]
    binary_targets = [1 if t > 0.5 else 0 for t in all_target_probs]

    f1 = f1_score(binary_targets, binary_preds, zero_division=0)
    precision = precision_score(binary_targets, binary_preds, zero_division=0)
    recall = recall_score(binary_targets, binary_preds, zero_division=0)

    print(f"   >>> [Metrics] F1: {f1:.4f} (Prec: {precision:.4f}, Rec: {recall:.4f}) | MSE: {mse_score:.5f}")