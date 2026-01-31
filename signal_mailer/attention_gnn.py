"""
Multi-head Attention GNN (v4.0)
자산 간 동적 관계를 학습하는 Attention 메커니즘 적용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphAttentionLayer(nn.Module):
    """Single Graph Attention Layer (GAT)"""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Linear transformations
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_features]
            adj: Adjacency matrix [N, N]
        Returns:
            Updated node features [N, out_features]
        """
        N = x.size(0)

        # Transform features
        h = self.W(x)  # [N, out_features]

        # Prepare attention inputs
        h_repeat = h.repeat(N, 1, 1)  # [N, N, out_features]
        h_repeat_interleave = h.repeat_interleave(N, dim=0).view(
            N, N, -1
        )  # [N, N, out_features]

        # Concatenate for attention
        attention_input = torch.cat(
            [h_repeat_interleave, h_repeat], dim=2
        )  # [N, N, 2*out_features]

        # Calculate attention coefficients
        e = self.leaky_relu(self.a(attention_input).squeeze(-1))  # [N, N]

        # Mask non-adjacent nodes
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        # Softmax
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        # Aggregate
        h_prime = torch.mm(attention, h)  # [N, out_features]

        return h_prime


class MultiHeadAttentionGNN(nn.Module):
    """Multi-head Graph Attention Network (v4.0)

    논문의 Cross-Asset Attention을 구현한 모델.
    여러 개의 attention head가 다양한 관점에서 자산 간 관계를 학습.
    """

    def __init__(
        self,
        in_features: int = 10,
        hidden_features: int = 16,
        out_features: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super(MultiHeadAttentionGNN, self).__init__()

        self.num_heads = num_heads

        # Multi-head attention (first layer)
        self.attention_heads = nn.ModuleList(
            [
                GraphAttentionLayer(in_features, hidden_features, dropout)
                for _ in range(num_heads)
            ]
        )

        # Output layer (aggregates multi-head outputs)
        self.out_layer = nn.Linear(hidden_features * num_heads, out_features)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_features]
            adj: Adjacency matrix [N, N]
        Returns:
            Node scores [N, out_features]
        """
        # Apply each attention head
        head_outputs = []
        for attention in self.attention_heads:
            h = attention(x, adj)
            head_outputs.append(h)

        # Concatenate all heads
        h_concat = torch.cat(head_outputs, dim=1)  # [N, hidden * num_heads]

        # Apply dropout and output layer
        h_concat = self.dropout(h_concat)
        out = self.out_layer(h_concat)  # [N, out_features]

        return out


# Backward compatibility alias
class AttentionGCN(MultiHeadAttentionGNN):
    """Alias for backward compatibility"""

    pass


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-head Attention GNN 테스트")
    print("=" * 60)

    # Test configuration
    num_nodes = 9  # 9 tickers
    in_features = 10  # 10 technical indicators
    hidden_features = 16
    out_features = 1
    num_heads = 4

    # Create model
    model = MultiHeadAttentionGNN(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        num_heads=num_heads,
    )

    print(f"\n모델 구조:")
    print(f"  입력 차원: {in_features}")
    print(f"  Hidden 차원: {hidden_features}")
    print(f"  Attention Heads: {num_heads}")
    print(f"  출력 차원: {out_features}")

    # Test forward pass
    x = torch.randn(num_nodes, in_features)
    adj = torch.eye(num_nodes)  # Simple identity adjacency

    model.eval()
    with torch.no_grad():
        output = model(x, adj)

    print(f"\n테스트 결과:")
    print(f"  입력 shape: {x.shape}")
    print(f"  출력 shape: {output.shape}")
    print(f"  출력 값: {output.squeeze().tolist()}")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n총 파라미터 수: {total_params:,}")

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)
