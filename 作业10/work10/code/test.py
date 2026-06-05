import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)


# 实现sinusoidal position encoding
def sinusoidal_position_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)
    return PE


def visualize_sinusoidal_pe(seq_len=50, d_model=128):
    pe = sinusoidal_position_encoding(seq_len, d_model)
    plt.figure(figsize=(12, 6))
    plt.imshow(pe, aspect="auto", cmap="viridis")
    plt.colorbar(label="PE value")
    plt.xlabel("Embedding dimension")
    plt.ylabel("Position")
    plt.title("Sinusoidal Position Encoding")
    plt.savefig(
        "/home/shi_chou_chu_jin/cv-course/work10/photo/sinusoidal_pe.png", dpi=150
    )
    plt.show()

    plt.figure(figsize=(12, 6))
    for dim in range(4):
        plt.plot(pe[:, dim], label=f"dim={dim}")
    plt.xlabel("Position")
    plt.ylabel("PE value")
    plt.legend()
    plt.title("Sinusoidal PE curves for first 4 dimensions")
    plt.grid(True)
    plt.savefig(
        "/home/shi_chou_chu_jin/cv-course/work10/photo/sinusoidal_pe_curves.png",
        dpi=150,
    )
    plt.show()


# 实现二维向量旋转
def rotate_2d(vector, theta):
    if isinstance(vector, np.ndarray):
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return np.dot(vector, R.T) if vector.ndim == 2 else np.dot(R, vector)
    else:
        R = torch.tensor(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=vector.dtype,
        )
        if vector.dim() == 2:
            return torch.mm(vector, R.T)
        else:
            return torch.mm(R, vector.unsqueeze(1)).squeeze(1)


def test_2d_rotation():
    print("\n实现二维向量旋转:")
    v = np.array([1.0, 0.0])
    theta = np.pi / 2
    v_rot = rotate_2d(v, theta)
    print(f"原始向量:{v}")
    print(f"旋转90°后:{v_rot} (期望[0,1])")

    angles = np.linspace(0, 2 * np.pi, 8)
    origin = np.array([0, 0])
    plt.figure(figsize=(6, 6))
    for theta in angles:
        v_rot = rotate_2d(v, theta)
        plt.arrow(
            origin[0],
            origin[1],
            v_rot[0],
            v_rot[1],
            head_width=0.05,
            head_length=0.05,
            fc="blue",
            ec="blue",
            alpha=0.7,
        )
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.grid(True)
    plt.title("2D Rotation of vector (1,0)")
    plt.axis("equal")
    plt.savefig(
        "/home/shi_chou_chu_jin/cv-course/work10/photo/2d_rotation.png", dpi=150
    )
    plt.show()


# 实现高维RoPE
def precompute_rope_freqs(d_model, max_seq_len, base=10000.0):
    assert d_model % 2 == 0, "d_model must be even"
    i = torch.arange(0, d_model, 2)
    theta = 1.0 / (base ** (i / d_model))
    positions = torch.arange(max_seq_len).float().unsqueeze(1)
    freqs = positions * theta.unsqueeze(0)
    return freqs


def apply_rope_batch_first(x, freqs):
    batch, seq_len, d_model = x.shape
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    theta = freqs[:seq_len, :]
    cos_theta = torch.cos(theta).unsqueeze(0)
    sin_theta = torch.sin(theta).unsqueeze(0)
    x1_rot = x1 * cos_theta - x2 * sin_theta
    x2_rot = x1 * sin_theta + x2 * cos_theta
    x_rot = torch.zeros_like(x)
    x_rot[..., 0::2] = x1_rot
    x_rot[..., 1::2] = x2_rot
    return x_rot


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=512, base=10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.register_buffer("freqs", precompute_rope_freqs(d_model, max_seq_len, base))

    def forward(self, x):
        return apply_rope_batch_first(x, self.freqs)


# 对比E+pos和RoPE的输入方式
def compare_embedding_methods():
    print("\n对比E+pos和RoPE的输入方式:")
    d_model = 8
    seq_len = 4
    batch = 2
    token_embed = torch.randn(batch, seq_len, d_model)
    pos_enc = torch.tensor(
        sinusoidal_position_encoding(seq_len, d_model), dtype=torch.float32
    )
    pos_enc = pos_enc.unsqueeze(0)
    add_input = token_embed + pos_enc
    rope_layer = RotaryPositionalEmbedding(d_model, max_seq_len=seq_len)
    rope_input = rope_layer(token_embed)
    print(f"Token embedding shape:{token_embed.shape}")
    print(f"Additive PE shape:{pos_enc.shape}")
    print(f"RoPE output shape:{rope_input.shape}")
    token_norm = torch.norm(token_embed, dim=-1)
    rope_norm = torch.norm(rope_input, dim=-1)
    max_diff = torch.max(torch.abs(token_norm - rope_norm)).item()
    print(f"\n模长不变性验证:token模长 vs RoPE模长 差异={max_diff:.6f}(应接近0)")
    print("\n分析:")
    print(" 加法位置编码:token_embed + pos_enc → 内容和位置直接相加,无法分离")
    print(" RoPE:旋转token嵌入,不改变模长,保留原始内容信息,同时引入位置依赖")
    return add_input, rope_input


# 用数值实验验证RoPE的相对位置性质
def verify_rope_relative_property():
    print("\n用数值实验验证RoPE的相对位置性质:")
    d_model = 32
    seq_len = 6
    batch = 1
    torch.manual_seed(123)
    q = torch.randn(batch, seq_len, d_model)
    k = torch.randn(batch, seq_len, d_model)
    rope_layer = RotaryPositionalEmbedding(d_model, max_seq_len=seq_len)
    q_rope = rope_layer(q)
    k_rope = rope_layer(k)
    scores_raw = torch.matmul(q, k.transpose(-2, -1))[0]
    scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))[0]

    rel_distances = [1, 2, 3]
    print("检查相同相对距离的attention score是否相等:")
    for d in rel_distances:
        values = []
        for i in range(seq_len - d):
            score = scores_rope[i, i + d].item()
            values.append(score)
        print(f"  相对距离 = {d}:scores = {values}")
        if len(values) > 1:
            std = np.std(values)
            print(f"    标准差 = {std:.6f} (越小说明相对位置性质越好)")

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(scores_raw, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Raw Q·K (no position)")
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    plt.subplot(1, 3, 2)
    plt.imshow(scores_rope, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("RoPE Q·K (with relative position)")
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    plt.subplot(1, 3, 3)
    diff = scores_rope - scores_raw
    plt.imshow(diff, cmap="seismic", vmin=-0.5, vmax=0.5)
    plt.colorbar()
    plt.title("RoPE - Raw (position effect)")
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    plt.tight_layout()
    plt.savefig(
        "/home/shi_chou_chu_jin/cv-course/work10/photo/rope_relative_property.png",
        dpi=150,
    )
    plt.show()

    freqs = precompute_rope_freqs(d_model, max_seq_len=seq_len)
    d = 2
    angles_diff = []
    for i in range(seq_len - d):
        diff_angle = (i + d - i) * freqs[0]
        angles_diff.append(diff_angle)
    first = angles_diff[0]
    all_same = all(torch.allclose(first, a) for a in angles_diff)
    print(
        f"\n验证旋转角度的差只依赖于相对距离:相对距离{d}的角度差是否一致 -> {all_same}"
    )
    return scores_raw, scores_rope


visualize_sinusoidal_pe(seq_len=50, d_model=128)
test_2d_rotation()
rope_layer = RotaryPositionalEmbedding(d_model=8, max_seq_len=10)
x = torch.randn(2, 5, 8)
x_rot = rope_layer(x)
print(f"用数值实验验证RoPE的相对位置性质:\n输入shape:{x.shape},输出shape:{x_rot.shape}")
compare_embedding_methods()
verify_rope_relative_property()
