import torch
import torch.nn as nn
import torch.nn.functional as F


def _act(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "gelu":
        return nn.GELU()
    elif name == "silu":
        return nn.SiLU(inplace=True)
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation: {name}")

class MLP(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=1280, output_ch=1280, resolution=1, nonlinearity="relu"):
        super(MLP, self).__init__()
        output_dim=output_ch*resolution*resolution
        self.resolution=resolution
        self.output_ch=output_ch
        self.fc1 = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x, x_ts):
        x = x.to(self.fc1.weight.dtype)
        x = self.fc1(x)
        return x.view(x.shape[0], self.output_ch, self.resolution, self.resolution)


class MLP2(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=640, output_ch=1280, resolution=1, nonlinearity="relu"):
        super(MLP2, self).__init__()
        output_dim=output_ch*resolution*resolution
        self.resolution=resolution
        self.output_ch=output_ch
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)
    def forward(self, x, x_ts):
        x = x.to(self.fc1.weight.dtype)
        x = self.fc1(x)
        x = self.fc2(x)
        return x.view(x.shape[0], self.output_ch, self.resolution, self.resolution)

class FiLMMap(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=1280, output_ch=1280, resolution=1,
                 reduction=0.5, nonlinearity="gelu", use_pair=False):
        super().__init__()
        self.resolution = resolution
        self.use_pair = use_pair
        out_ch = output_ch * (2 if use_pair else 1)
        out_dim = out_ch * resolution * resolution
        mid = max(1, int(hidden_dim * reduction))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mid, bias=True),
            _act(nonlinearity),
            nn.Linear(mid, out_dim, bias=True)
        )
        # γ 초기 안정화를 위한 편향 보정 옵션
        self.use_pair = use_pair
        self.output_ch = out_ch

    def forward(self, x, x_ts):
        y = self.mlp(x.to(next(self.mlp.parameters()).dtype))
        y = y.view(y.shape[0], self.output_ch, self.resolution, self.resolution)
        return y  # [B, C(또는 2C), H, W]; γ,β 분리는 호출부에서 C 단위로 split



class ConvAdapterMap(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=1280, output_ch=1280, resolution=8,
                 reduction=0.25, nonlinearity="gelu", drop=0.0):
        super().__init__()
        self.resolution = resolution
        self.output_ch = output_ch
        base_dim = output_ch * resolution * resolution

        # 먼저 coarse 맵 생성
        self.fc = nn.Linear(input_dim, base_dim, bias=True)

        # 공간 정제 블록
        rC = max(1, int(output_ch * reduction))
        self.norm = nn.GroupNorm(32, output_ch)
        self.conv1 = nn.Conv2d(output_ch, rC, kernel_size=1, bias=True)
        self.act = _act(nonlinearity)
        self.conv2 = nn.Conv2d(rC, output_ch, kernel_size=1, bias=True)
        self.gate = nn.Conv2d(output_ch, output_ch, kernel_size=1, bias=True)
        self.dropout = nn.Dropout2d(p=drop) if drop > 0 else nn.Identity()

        # 게이트를 작은 값으로 시작
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)

    def forward(self, x, x_ts):
        dtype = self.fc.weight.dtype
        y = self.fc(x.to(dtype)).view(x.shape[0], self.output_ch, self.resolution, self.resolution)
        # 1x1 병목 잔차
        z = self.norm(y)
        z = self.conv2(self.act(self.conv1(z)))
        z = self.dropout(z)
        g = torch.sigmoid(self.gate(y))
        y = y + g * z
        return y


class TokenMixerMap(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=1280, output_ch=1280, resolution=8,
                 num_tokens=8, key_dim=256, nonlinearity="gelu"):
        super().__init__()
        self.resolution = resolution
        self.output_ch = output_ch
        H = W = resolution
        N = H * W
        self.num_tokens = num_tokens

        # 스타일 토큰 생성: [B, K, key_dim]
        self.token_mlp = nn.Sequential(
            nn.Linear(input_dim, num_tokens * key_dim, bias=True),
            _act(nonlinearity)
        )

        # 위치 쿼리: [N, q_dim]
        q_dim = key_dim
        self.pos_query = nn.Parameter(torch.randn(N, q_dim) * 0.02)

        # K,V 생성
        self.k_proj = nn.Linear(key_dim, q_dim, bias=False)
        self.v_proj = nn.Linear(key_dim, output_ch, bias=False)

        # 혼합 후 1x1 정제
        self.post = nn.Sequential(
            nn.GroupNorm(32, output_ch),
            nn.Conv2d(output_ch, output_ch, kernel_size=1, bias=True)
        )

    def forward(self, x, x_ts):
        B = x.shape[0]
        dtype = self.k_proj.weight.dtype

        # 스타일 토큰
        t = self.token_mlp(x.to(dtype)).view(B, self.num_tokens, -1)         # [B,K,D]
        K = self.k_proj(t)                                                    # [B,K,D]
        V = self.v_proj(t)                                                    # [B,K,C]

        # 위치 쿼리
        Q = self.pos_query.to(dtype).unsqueeze(0).expand(B, -1, -1)           # [B,N,D]

        # 점수 및 가중합
        attn = torch.softmax((Q @ K.transpose(1, 2)) / (K.shape[-1] ** 0.5), dim=-1)  # [B,N,K]
        Y = attn @ V                                                          # [B,N,C]
        Y = Y.transpose(1, 2).view(B, self.output_ch, self.resolution, self.resolution)
        Y = self.post(Y)
        return Y

class MoEStyleMap(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=1280, output_ch=1280, resolution=8,
                 num_experts=4, reduction=0.25, nonlinearity="gelu"):
        super().__init__()
        self.resolution = resolution
        self.output_ch = output_ch
        self.num_experts = num_experts
        out_dim = output_ch * resolution * resolution
        mid = max(1, int(hidden_dim * reduction))

        # 게이팅
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            _act(nonlinearity),
            nn.Linear(hidden_dim, num_experts, bias=True)
        )

        # 전문가들: 작고 얕은 MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, mid, bias=True),
                _act(nonlinearity),
                nn.Linear(mid, out_dim, bias=True)
            ) for _ in range(num_experts)
        ])

        # 초기에는 평균 동작
        for e in self.experts:
            nn.init.normal_(e[-1].weight, std=0.01)
            nn.init.zeros_(e[-1].bias)

    def forward(self, x, x_ts):
        dtype = self.experts[0][0].weight.dtype
        x = x.to(dtype)
        logits = self.gate(x)                 # [B,K]
        w = torch.softmax(logits, dim=-1)     # [B,K]

        outs = []
        for e in self.experts:
            y = e(x)                          # [B, out_dim]
            outs.append(y.unsqueeze(1))       # [B,1,out_dim]
        Y = torch.sum(torch.cat(outs, dim=1) * w.unsqueeze(-1), dim=1)  # [B,out_dim]
        Y = Y.view(x.shape[0], self.output_ch, self.resolution, self.resolution)
        return Y




model_types = {
    "MLP": MLP,
    "MLP2": MLP2,
    "FiLMMap": FiLMMap,              # γ,β 맵 생성(옵션으로 두 맵 concat)
    "ConvAdapterMap": ConvAdapterMap,  # 1x1 병목 정제 포함
    "TokenMixerMap": TokenMixerMap,  # 위치 쿼리로 K 스타일 토큰 혼합
    "MoEStyleMap": MoEStyleMap,      # 전문가 혼합
}