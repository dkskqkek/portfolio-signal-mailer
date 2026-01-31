"""
GNN 모델 가중치 재초기화 (v3.2: 10 features)
기존 2개 입력에서 10개 입력으로 확장
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

GNN_DATA_DIR = r"d:\gg\data\gnn"
WEIGHT_FILE = os.path.join(GNN_DATA_DIR, "gnn_weights.pth")
WEIGHT_FILE_BACKUP = os.path.join(GNN_DATA_DIR, "gnn_weights_v3_backup.pth")


class SimpleGCN(nn.Module):
    """v3.2: 10 input features"""

    def __init__(self, in_features=10, hidden_features=16, out_features=1):
        super(SimpleGCN, self).__init__()
        self.conv1 = nn.Linear(in_features, hidden_features)
        self.conv2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, adj):
        x = torch.mm(adj, x)
        x = F.relu(self.conv1(x))
        x = torch.mm(adj, x)
        x = self.conv2(x)
        return x


if __name__ == "__main__":
    print("=" * 60)
    print("GNN 모델 가중치 재초기화 (v3.2: 10 features)")
    print("=" * 60)

    # 1. 기존 가중치 백업
    if os.path.exists(WEIGHT_FILE):
        import shutil

        shutil.copy(WEIGHT_FILE, WEIGHT_FILE_BACKUP)
        print(f"\n[1] 기존 가중치 백업 완료: {WEIGHT_FILE_BACKUP}")

    # 2. 새 모델 생성 (10 features)
    model = SimpleGCN(in_features=10, hidden_features=16, out_features=1)

    # Xavier 초기화 (안정적인 학습을 위해)
    nn.init.xavier_uniform_(model.conv1.weight)
    nn.init.xavier_uniform_(model.conv2.weight)
    nn.init.zeros_(model.conv1.bias)
    nn.init.zeros_(model.conv2.bias)

    print(f"\n[2] 새 모델 생성 완료:")
    print(f"  - 입력 차원: 10 (기존 2)")
    print(f"  - Hidden 차원: 16")
    print(f"  - 출력 차원: 1")

    # 3. 가중치 저장
    torch.save(model.state_dict(), WEIGHT_FILE)
    print(f"\n[3] 새 가중치 저장 완료: {WEIGHT_FILE}")

    # 4. 검증
    loaded_model = SimpleGCN(in_features=10, hidden_features=16, out_features=1)
    loaded_model.load_state_dict(torch.load(WEIGHT_FILE, weights_only=True))
    loaded_model.eval()

    # 테스트 입력 (9개 노드, 10개 특성)
    test_input = torch.randn(9, 10)
    test_adj = torch.eye(9)  # 단위 행렬

    with torch.no_grad():
        output = loaded_model(test_input, test_adj)

    print(f"\n[4] 검증 완료:")
    print(f"  - 입력 shape: {test_input.shape}")
    print(f"  - 출력 shape: {output.shape}")
    print(f"  - 출력 샘플: {output.squeeze().tolist()[:3]}...")

    print("\n" + "=" * 60)
    print("GNN v3.2 모델 준비 완료!")
    print("=" * 60)
