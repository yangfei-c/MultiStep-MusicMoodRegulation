from collections.abc import Sequence

import torch
from torch import nn


class MaskedMeanStdPooling(nn.Module):
    """计算有效 segment 的均值和总体标准差。"""

    def __init__(self, eps: float = 1.0e-5) -> None:
        super().__init__()
        if eps <= 0: raise ValueError(f"eps 必须大于 0，实际为 {eps}")
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3: raise ValueError(f"x 应为 [B,S,H]，实际为 {tuple(x.shape)}")
        if mask.shape != x.shape[:2] or mask.dtype != torch.bool: raise ValueError(f"mask 应为 bool 类型的 {tuple(x.shape[:2])}，实际为 {tuple(mask.shape)}、{mask.dtype}")
        if (mask.sum(1) == 0).any().item(): raise ValueError("存在没有有效 segment 的歌曲")

        mask = mask.unsqueeze(-1).to(x.dtype)
        count = mask.sum(1)
        mean = (x * mask).sum(1) / count
        std = (((x - mean.unsqueeze(1)).square() * mask).sum(1) / count + self.eps).sqrt()
        return torch.cat((mean, std), dim=-1)


class EnhancedBaselineModel(nn.Module):
    """固定 MERT 层拼接、mean+std pooling 和四个任务头。"""

    def __init__(self, layer_indices: Sequence[int] = (5, 6), hidden_dim: int = 256, dropout: float = 0.2, pooling_eps: float = 1.0e-5) -> None:
        super().__init__()
        indices = tuple(map(int, layer_indices))

        if not indices or len(indices) != len(set(indices)): raise ValueError(f"layer_indices 为空或存在重复：{indices}")
        if any(index < 0 or index >= 12 for index in indices): raise ValueError(f"layer_indices 必须位于 [0,11]，实际为 {indices}")
        if hidden_dim <= 0: raise ValueError(f"hidden_dim 必须大于 0，实际为 {hidden_dim}")
        if not 0 <= dropout < 1: raise ValueError(f"dropout 必须位于 [0,1)，实际为 {dropout}")

        self.hidden_dim = hidden_dim
        self.register_buffer("layer_indices", torch.tensor(indices, dtype=torch.long), persistent=False)

        self.segment_projection = nn.Sequential(nn.LayerNorm(len(indices) * 768), nn.Linear(len(indices) * 768, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.song_pooling = MaskedMeanStdPooling(pooling_eps)
        self.song_fusion = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.genre_head = nn.Linear(hidden_dim, 87)
        self.instrument_head = nn.Linear(hidden_dim, 40)
        self.mood_head = nn.Linear(hidden_dim, 56)
        self.va_head = nn.Linear(hidden_dim, 2)

    @staticmethod
    def check_inputs(features: torch.Tensor, mask: torch.Tensor) -> None:
        if features.ndim != 4 or tuple(features.shape[2:]) != (12, 768): raise ValueError(f"features 应为 [B,S,12,768]，实际为 {tuple(features.shape)}")
        if not features.is_floating_point(): raise TypeError(f"features 应为浮点类型，实际为 {features.dtype}")
        if mask.shape != features.shape[:2] or mask.dtype != torch.bool: raise ValueError(f"mask 应为 bool 类型的 {tuple(features.shape[:2])}，实际为 {tuple(mask.shape)}、{mask.dtype}")
        if (mask.sum(1) == 0).any().item(): raise ValueError("存在没有有效 segment 的歌曲")

    def encode(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self.check_inputs(features, mask)
        segments = features.index_select(2, self.layer_indices).flatten(2)
        return self.song_fusion(self.song_pooling(self.segment_projection(segments), mask))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.encode(features, mask)
        genre, instrument, mood = self.genre_head(embedding), self.instrument_head(embedding), self.mood_head(embedding)
        return {
            "song_embedding": embedding,
            "genre_logits": genre,
            "instrument_logits": instrument,
            "mood_logits": mood,
            "tag_logits": torch.cat((genre, instrument, mood), dim=-1),
            "va_predictions": self.va_head(embedding),
        }