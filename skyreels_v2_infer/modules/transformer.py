# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SkyReels-V2 核心Transformer模型实现
# 这是整个视频生成模型的大脑，负责理解文本条件并生成视频特征

import math
import numpy as np
import torch
import torch.amp as amp  # 混合精度训练支持
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from torch.backends.cuda import sdp_kernel
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import flex_attention

from .attention import flash_attention  # 引入高效的Flash Attention实现


flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune")

DISABLE_COMPILE = False  # 控制是否禁用PyTorch编译优化

__all__ = ["WanModel"]  # 对外暴露的主要类


def sinusoidal_embedding_1d(dim, position):
    """
    生成1维正弦位置编码，用于时间步嵌入
    
    Args:
        dim: 嵌入维度，必须是偶数（因为要分别计算sin和cos）
        position: 位置索引，通常是时间步数值
    
    Returns:
        正弦余弦位置编码 [position_num, dim]
    """
    # 预处理：确保维度是偶数
    assert dim % 2 == 0
    half = dim // 2  # 一半用于sin，一半用于cos
    position = position.type(torch.float64)

    # 计算正弦余弦编码
    # 使用不同频率的正弦波来编码位置信息，频率随维度递减
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast("cuda", enabled=False)  # 禁用自动混合精度，保证数值稳定性
def rope_params(max_seq_len, dim, theta=10000):
    """
    生成RoPE（旋转位置编码）的频率参数
    
    Args:
        max_seq_len: 最大序列长度
        dim: 每个头的维度大小
        theta: 旋转频率的基数，默认10000
    
    Returns:
        复数形式的旋转频率矩阵，用于后续的旋转操作
    """
    assert dim % 2 == 0  # 维度必须是偶数
    # 计算每个维度对应的旋转频率，频率随维度递减
    freqs = torch.outer(
        torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim))
    )
    # 转换为复数极坐标形式，便于旋转计算
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast("cuda", enabled=False)  # 禁用自动混合精度，保证数值稳定性
def rope_apply(x, grid_sizes, freqs):
    """
    应用3D RoPE位置编码到输入张量
    这是SkyReels-V2的核心创新：将RoPE扩展到3维（时间+2D空间）
    
    Args:
        x: 输入张量 [batch, seq_len, heads, head_dim]
        grid_sizes: 网格尺寸 [时间帧数F, 高度H, 宽度W]  
        freqs: 旋转频率参数
    
    Returns:
        应用了3D位置编码的张量
    """
    n, c = x.size(2), x.size(3) // 2  # n=注意力头数, c=每头维度的一半
    bs = x.size(0)  # batch_size

    # 将频率分割为三个部分：时间、高度、宽度维度
    # 分配原则：时间维度占大部分，高度和宽度各占1/3
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # 获取网格尺寸：帧数、高度、宽度
    f, h, w = grid_sizes.tolist()
    seq_len = f * h * w  # 总序列长度

    # 将输入转换为复数形式，便于旋转操作
    x = torch.view_as_complex(x.to(torch.float32).reshape(bs, seq_len, n, -1, 2))
    
    # 为每个空间位置计算对应的旋转频率
    freqs_i = torch.cat(
        [
            # 时间维度的频率：每帧都有对应的时间编码
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            # 高度维度的频率：每行都有对应的高度编码  
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            # 宽度维度的频率：每列都有对应的宽度编码
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(seq_len, 1, -1)

    # 应用旋转编码：复数乘法实现旋转
    x = torch.view_as_real(x * freqs_i).flatten(3)

    return x


@torch.compile(dynamic=True, disable=DISABLE_COMPILE)
def fast_rms_norm(x, weight, eps):
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    x = x.type_as(x) * weight
    return x


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return fast_rms_norm(x, self.weight, self.eps)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self._flag_ar_attention = False

    def set_ar_attention(self):
        self._flag_ar_attention = True

    def forward(self, x, grid_sizes, freqs, block_mask):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        x = x.to(self.q.weight.dtype)
        q, k, v = qkv_fn(x)

        if not self._flag_ar_attention:
            q = rope_apply(q, grid_sizes, freqs)
            k = rope_apply(k, grid_sizes, freqs)
            x = flash_attention(q=q, k=k, v=v, window_size=self.window_size)
        else:
            q = rope_apply(q, grid_sizes, freqs)
            k = rope_apply(k, grid_sizes, freqs)
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)

            with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                x = (
                    torch.nn.functional.scaled_dot_product_attention(
                        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=block_mask
                    )
                    .transpose(1, 2)
                    .contiguous()
                )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    def forward(self, x, context):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img)
        # compute attention
        x = flash_attention(q, k, v)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


def mul_add(x, y, z):
    return x.float() + y.float() * z.float()


def mul_add_add(x, y, z):
    return x.float() * (1 + y) + z


mul_add_compile = torch.compile(mul_add, dynamic=True, disable=DISABLE_COMPILE)
mul_add_add_compile = torch.compile(mul_add_add, dynamic=True, disable=DISABLE_COMPILE)


class WanAttentionBlock(nn.Module):
    """
    WanAttentionBlock: SkyReels-V2的核心构建块
    这是Transformer的基本单元，每个模型有32个这样的Block
    每个Block包含：自注意力 + 交叉注意力 + 前馈网络
    """
    def __init__(
        self,
        cross_attn_type,    # 交叉注意力类型："t2v_cross_attn"或"i2v_cross_attn"
        dim,                # 模型隐藏维度（token的特征维度），14B模型是2048
        ffn_dim,            # 前馈网络中间层维度，通常是dim的4倍（8192）
        num_heads,          # 多头注意力的头数，14B模型是16个头
        window_size=(-1, -1),  # 窗口注意力大小，(-1,-1)表示全局注意力
        qk_norm=True,       # 是否对Query和Key进行归一化，提高训练稳定性
        cross_attn_norm=False,  # 是否对交叉注意力进行额外归一化
        eps=1e-6,          # 归一化的小值，防止除零
    ):
        super().__init__()
        # 保存配置参数
        self.dim = dim                              # 隐藏维度：每个token有多少个特征
        self.ffn_dim = ffn_dim                      # 前馈网络维度：中间层的神经元数量
        self.num_heads = num_heads                  # 注意力头数：多头注意力的并行数量
        self.window_size = window_size              # 注意力窗口：控制每个token能看到多远
        self.qk_norm = qk_norm                      # QK归一化：是否标准化Query和Key
        self.cross_attn_norm = cross_attn_norm      # 交叉注意力归一化开关
        self.eps = eps                              # 数值稳定性参数

        # ===== 核心网络层定义 =====
        # norm1: 自注意力前的层归一化，用于稳定训练和提升性能
        self.norm1 = WanLayerNorm(dim, eps)
        
        # self_attn: 自注意力机制，让视频帧之间相互交流信息
        # 每个像素点都能看到其他像素点，学习时空依赖关系
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        
        # norm3: 交叉注意力前的层归一化（如果启用cross_attn_norm的话）
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        
        # cross_attn: 交叉注意力机制，让视频特征与文本条件进行交互
        # 这里是视频理解文本指令的关键：视频问文本"你想让我生成什么？"
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)
        
        # norm2: 前馈网络前的层归一化
        self.norm2 = WanLayerNorm(dim, eps)
        
        # ffn: 前馈神经网络，对每个位置的特征进行非线性变换
        # 结构：dim -> ffn_dim -> dim，中间使用GELU激活函数
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

        # ===== 调制参数 =====
        # modulation: 可学习的调制参数，用于动态调整特征
        # 形状[1, 6, dim]对应6个不同的调制用途（自注意力前后、交叉注意力前后、FFN前后）
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def set_ar_attention(self):
        self.self_attn.set_ar_attention()

    def forward(
        self,
        x,          # 输入的视频特征张量 [batch, sequence_len, hidden_dim]
        e,          # 时间嵌入张量 [batch, 6, hidden_dim] 或 [batch, 6, seq_len, hidden_dim]
        grid_sizes, # 视频网格尺寸 [帧数F, 高度H, 宽度W]
        freqs,      # RoPE旋转频率参数
        context,    # 文本条件特征 [batch, text_len, text_dim]
        block_mask, # 注意力掩码，用于Diffusion Forcing
    ):
        """
        WanAttentionBlock前向传播：Transformer Block的核心执行流程
        
        处理流程：
        1. 时间调制参数处理
        2. 自注意力：视频内部信息交流
        3. 交叉注意力：视频与文本条件交互  
        4. 前馈网络：非线性特征变换
        
        Args:
            x: 视频特征张量 [batch_size, sequence_length, hidden_dim]
               sequence_length = 帧数 × 高度patch数 × 宽度patch数
            e: 时间嵌入张量，包含6组调制参数用于不同阶段
            grid_sizes: 视频空间网格 [帧数F, 高度H, 宽度W]  
            freqs: RoPE位置编码的旋转频率
            context: 文本条件特征
            block_mask: 因果掩码（Diffusion Forcing用）
        """
        
        # ===== 第1步：处理时间调制参数 =====
        # 时间嵌入e包含6组参数，分别用于调制不同阶段的计算
        if e.dim() == 3:  # 标准情况：[batch, 6, dim]
            modulation = self.modulation  # [1, 6, dim] 可学习的调制基准
            with amp.autocast("cuda", dtype=torch.float32):
                # 将基准调制参数与时间嵌入相加，然后拆分为6组
                e = (modulation + e).chunk(6, dim=1)
        elif e.dim() == 4:  # Diffusion Forcing模式：[batch, 6, seq_len, dim]
            modulation = self.modulation.unsqueeze(2)  # [1, 6, 1, dim]
            with amp.autocast("cuda", dtype=torch.float32):
                e = (modulation + e).chunk(6, dim=1)
            e = [ei.squeeze(1) for ei in e]  # 移除多余维度

        # ===== 第2步：自注意力计算 =====
        # 让视频的每个位置都能看到其他位置，学习时空依赖关系
        # mul_add_add_compile: 编译优化的 x * (1 + e[1]) + e[0] 操作
        out = mul_add_add_compile(self.norm1(x), e[1], e[0])  # 归一化后用时间嵌入调制
        y = self.self_attn(out, grid_sizes, freqs, block_mask)  # 应用自注意力
        with amp.autocast("cuda", dtype=torch.float32):
            x = mul_add_compile(x, y, e[2])  # 残差连接：x + y * e[2]

        # ===== 第3步：交叉注意力 + 前馈网络 =====
        # 定义内部函数处理后续步骤
        def cross_attn_ffn(x, context, e):
            """交叉注意力和前馈网络的组合处理"""
            dtype = context.dtype
            
            # 交叉注意力：视频特征向文本条件"提问"
            # "我应该生成什么样的内容？"
            x = x + self.cross_attn(self.norm3(x.to(dtype)), context)
            
            # 前馈网络：对特征进行非线性变换，增强表达能力
            # 先归一化并用时间嵌入调制，再通过FFN
            y = self.ffn(mul_add_add_compile(self.norm2(x), e[4], e[3]).to(dtype))
            with amp.autocast("cuda", dtype=torch.float32):
                x = mul_add_compile(x, y, e[5])  # 残差连接
            return x

        # 执行交叉注意力和前馈网络
        x = cross_attn_ffn(x, context, e)
        
        # 转换为bfloat16格式输出，节省显存
        return x.to(torch.bfloat16)


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        with amp.autocast("cuda", dtype=torch.float32):
            if e.dim() == 2:
                modulation = self.modulation  # 1, 2, dim
                e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)

            elif e.dim() == 3:
                modulation = self.modulation.unsqueeze(2)  # 1, 2, seq, dim
                e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
                e = [ei.squeeze(1) for ei in e]
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    WanModel: SkyReels-V2的核心Transformer模型
    
    这是整个视频生成系统的"大脑"，负责：
    1. 理解文本指令和图像条件
    2. 在潜在空间中生成视频特征
    3. 支持T2V（文本生视频）和I2V（图像生视频）两种模式
    4. 支持Diffusion Forcing技术实现无限长视频生成
    
    模型规模：
    - 14B版本：32层，2048隐藏维度，16个注意力头
    - 1.3B版本：更小的配置，适合资源有限的环境
    """

    # 配置相关的忽略字段和模块分割设置
    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim", "window_size"]
    _no_split_modules = ["WanAttentionBlock"]  # 不要分割WanAttentionBlock，保持完整性

    _supports_gradient_checkpointing = True  # 支持梯度检查点，节省显存

    @register_to_config
    def __init__(
        self,
        model_type="t2v",           # 模型类型："t2v"(文本生视频) 或 "i2v"(图像生视频)
        patch_size=(1, 2, 2),       # 3D patch大小：(时间, 高度, 宽度)，将视频分块为tokens
        text_len=512,               # 文本序列最大长度：支持的最大token数
        in_dim=16,                  # 输入维度：VAE潜在空间的通道数
        dim=2048,                   # 隐藏维度：每个token的特征维度（14B模型用2048）
        ffn_dim=8192,              # 前馈网络维度：FFN中间层大小，通常是dim的4倍
        freq_dim=256,              # 频率维度：时间嵌入的维度大小
        text_dim=4096,             # 文本特征维度：T5编码器输出的特征大小
        out_dim=16,                # 输出维度：预测的噪声通道数，与in_dim相同
        num_heads=16,              # 注意力头数：多头注意力机制的并行数量
        num_layers=32,             # Transformer层数：WanAttentionBlock的堆叠数量
        window_size=(-1, -1),      # 注意力窗口：(-1,-1)表示全局注意力，其他值表示局部窗口
        qk_norm=True,              # QK归一化：是否对Query和Key进行归一化，提升稳定性
        cross_attn_norm=True,      # 交叉注意力归一化：是否在交叉注意力前额外归一化
        inject_sample_info=False,  # 是否注入采样信息：如FPS等额外条件
        eps=1e-6,                  # 数值稳定性：归一化时的小值，防止除零错误
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ["t2v", "i2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.num_frame_per_block = 1
        self.flag_causal_attention = False
        self.block_mask = None
        self.enable_teacache = False

        # ===== 嵌入层：将不同模态转换为统一表示 =====
        
        # patch_embedding: 3D卷积，将视频分块并投影到hidden_dim空间
        # 输入：[batch, 16, T, H, W] (VAE潜在特征)
        # 输出：[batch, dim, T', H', W'] (T'=T/1, H'=H/2, W'=W/2)
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        
        # text_embedding: 文本特征投影网络，将T5输出转换为模型维度
        # T5输出4096维 -> 经过两层MLP -> 转换为2048维（与视频特征对齐）
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))

        # time_embedding: 时间步嵌入，将扩散时间步转换为特征
        # 输入：时间步数值 -> 正弦编码(256维) -> MLP(2048维)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        
        # time_projection: 将时间嵌入扩展为6组调制参数
        # 每组用于调制Transformer Block的不同阶段（自注意力前后、交叉注意力前后、FFN前后）
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # 可选：FPS等采样信息嵌入
        if inject_sample_info:
            # fps_embedding: 帧率嵌入，支持2种FPS类型（16fps和24fps）
            self.fps_embedding = nn.Embedding(2, dim)
            # fps_projection: 将FPS嵌入也扩展为6组调制参数，与时间嵌入相加
            self.fps_projection = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim * 6))

        # ===== Transformer主体：32个WanAttentionBlock堆叠 =====
        
        # 根据模型类型选择交叉注意力类型
        # t2v: 只处理文本条件 | i2v: 处理文本+图像条件
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        
        # blocks: 核心Transformer层，32个WanAttentionBlock组成深度网络
        # 每个Block包含：自注意力(视频内交流) + 交叉注意力(理解文本) + FFN(特征变换)
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
                for _ in range(num_layers)  # 14B模型: 32层，1.3B模型: 更少层数
            ]
        )

        # ===== 输出头：将特征转换回噪声预测 =====
        # head: 最终输出层，将Transformer特征转换为噪声预测
        # 输入：[batch, seq_len, dim] -> 输出：[batch, seq_len, out_dim*patch_volume]
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))],
            dim=1,
        )

        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim)

        self.gradient_checkpointing = False

        self.cpu_offloading = False

        self.inject_sample_info = inject_sample_info
        # initialize weights
        self.init_weights()

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def zero_init_i2v_cross_attn(self):
        print("zero init i2v cross attn")
        for i in range(self.num_layers):
            self.blocks[i].cross_attn.v_img.weight.data.zero_()
            self.blocks[i].cross_attn.v_img.bias.data.zero_()

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21, frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(start=0, end=total_length, step=frame_seqlen * num_frame_per_block, device=device)

        for tmp in frame_indices:
            ends[tmp : tmp + frame_seqlen * num_frame_per_block] = tmp + frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )

        return block_mask

    def initialize_teacache(self, enable_teacache=True, num_steps=25, teacache_thresh=0.15, use_ret_steps=False, ckpt_dir=''):
        self.enable_teacache = enable_teacache
        print('using teacache')
        self.cnt = 0
        self.num_steps = num_steps
        self.teacache_thresh = teacache_thresh
        self.accumulated_rel_l1_distance_even = 0
        self.accumulated_rel_l1_distance_odd = 0
        self.previous_e0_even = None
        self.previous_e0_odd = None
        self.previous_residual_even = None
        self.previous_residual_odd = None
        self.use_ref_steps = use_ret_steps
        if "I2V" in ckpt_dir:
            if use_ret_steps:
                if '540P' in ckpt_dir:
                    self.coefficients = [ 2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01]
                if '720P' in ckpt_dir:
                    self.coefficients = [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02]
                self.ret_steps = 5*2
                self.cutoff_steps = num_steps*2
            else:
                if '540P' in ckpt_dir:
                    self.coefficients = [-3.02331670e+02,  2.23948934e+02, -5.25463970e+01,  5.87348440e+00, -2.01973289e-01]
                if '720P' in ckpt_dir:
                    self.coefficients = [-114.36346466,   65.26524496,  -18.82220707,    4.91518089,   -0.23412683]
                self.ret_steps = 1*2
                self.cutoff_steps = num_steps*2 - 2
        else:
            if use_ret_steps:
                if '1.3B' in ckpt_dir:
                    self.coefficients = [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02]
                if '14B' in ckpt_dir:
                    self.coefficients = [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01]
                self.ret_steps = 5*2
                self.cutoff_steps = num_steps*2
            else:
                if '1.3B' in ckpt_dir:
                    self.coefficients = [2.39676752e+03, -1.31110545e+03,  2.01331979e+02, -8.29855975e+00, 1.37887774e-01]
                if '14B' in ckpt_dir:
                    self.coefficients = [-5784.54975374,  5449.50911966, -1811.16591783,   256.27178429, -13.02252404]
                self.ret_steps = 1*2
                self.cutoff_steps = num_steps*2 - 2

    def forward(self, x, t, context, clip_fea=None, y=None, fps=None):
        """
        WanModel前向传播：SkyReels-V2的核心推理流程
        
        整体流程：
        1. 输入预处理：视频patch化、设备同步
        2. 嵌入生成：时间嵌入、文本嵌入、位置编码
        3. Transformer处理：32层WanAttentionBlock逐层计算
        4. 输出生成：特征转换为噪声预测
        5. 后处理：unpatch化恢复视频形状

        Args:
            x: 输入视频张量 [batch, C_in=16, F, H, W] (VAE潜在特征)
            t: 扩散时间步 [batch] 或 [batch, F] (Diffusion Forcing模式)
            context: 文本条件特征 [batch, text_len, text_dim=4096] (来自T5编码器)
            clip_fea: CLIP图像特征 [batch, 257, 1280] (I2V模式用，可选)
            y: 条件视频输入 [batch, C_in, F, H, W] (I2V模式用，可选)
            fps: 帧率信息列表 (可选)

        Returns:
            预测的噪声张量 [batch, C_out=16, F, H/8, W/8] (用于扩散去噪)
        """
        # ===== 第1步：输入验证和预处理 =====
        if self.model_type == "i2v":
            # I2V模式需要图像特征和条件视频
            assert clip_fea is not None and y is not None
            
        # 设备同步：确保所有张量在同一设备上
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # I2V模式：将条件视频与输入视频拼接
        # x: [batch, 16, F, H, W], y: [batch, 16, F, H, W] -> [batch, 32, F, H, W]
        if y is not None:
            x = torch.cat([x, y], dim=1)

        # ===== 第2步：视频patch化和特征提取 =====
        # patch_embedding: 3D卷积将视频分块并转换为token序列
        # [batch, in_dim, F, H, W] -> [batch, dim, F', H', W']
        x = self.patch_embedding(x)
        grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long)  # 记录网格尺寸[F', H', W']
        # 转换为序列格式：[batch, dim, F', H', W'] -> [batch, sequence_length, dim]
        x = x.flatten(2).transpose(1, 2)  # sequence_length = F' * H' * W'

        if self.flag_causal_attention:
            frame_num = grid_sizes[0]
            height = grid_sizes[1]
            width = grid_sizes[2]
            block_num = frame_num // self.num_frame_per_block
            range_tensor = torch.arange(block_num).view(-1, 1)
            range_tensor = range_tensor.repeat(1, self.num_frame_per_block).flatten()
            casual_mask = range_tensor.unsqueeze(0) <= range_tensor.unsqueeze(1)  # f, f
            casual_mask = casual_mask.view(frame_num, 1, 1, frame_num, 1, 1).to(x.device)
            casual_mask = casual_mask.repeat(1, height, width, 1, height, width)
            casual_mask = casual_mask.reshape(frame_num * height * width, frame_num * height * width)
            self.block_mask = casual_mask.unsqueeze(0).unsqueeze(0)

        # time embeddings
        with amp.autocast("cuda", dtype=torch.float32):
            if t.dim() == 2:
                b, f = t.shape
                _flag_df = True
            else:
                _flag_df = False

            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(self.patch_embedding.weight.dtype)
            )  # b, dim
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))  # b, 6, dim

            if self.inject_sample_info:
                fps = torch.tensor(fps, dtype=torch.long, device=device)

                fps_emb = self.fps_embedding(fps).float()
                if _flag_df:
                    e0 = e0 + self.fps_projection(fps_emb).unflatten(1, (6, self.dim)).repeat(t.shape[1], 1, 1)
                else:
                    e0 = e0 + self.fps_projection(fps_emb).unflatten(1, (6, self.dim))

            if _flag_df:
                e = e.view(b, f, 1, 1, self.dim)
                e0 = e0.view(b, f, 1, 1, 6, self.dim)
                e = e.repeat(1, 1, grid_sizes[1], grid_sizes[2], 1).flatten(1, 3)
                e0 = e0.repeat(1, 1, grid_sizes[1], grid_sizes[2], 1, 1).flatten(1, 3)
                e0 = e0.transpose(1, 2).contiguous()

            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # ===== 第4步：文本条件处理 =====
        # 将文本特征从T5维度(4096)投影到模型维度(2048)
        context = self.text_embedding(context)

        # I2V模式：添加CLIP图像特征作为额外条件
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # [batch, 257, dim] CLIP图像token
            # 将图像token与文本token拼接：[图像token, 文本token]
            context = torch.concat([context_clip, context], dim=1)

        # ===== 第5步：准备Transformer输入参数 =====
        kwargs = dict(
            e=e0,                        # 时间嵌入调制参数 [batch, 6, dim]
            grid_sizes=grid_sizes,       # 视频网格尺寸 [F', H', W']
            freqs=self.freqs,           # RoPE旋转频率参数
            context=context,            # 文本(+图像)条件特征
            block_mask=self.block_mask  # 注意力掩码(Diffusion Forcing用)
        )
        if self.enable_teacache:
            modulated_inp = e0 if self.use_ref_steps else e
            # teacache
            if self.cnt%2==0: # even -> conditon
                self.is_even = True
                if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc_even = True
                    self.accumulated_rel_l1_distance_even = 0
                else:
                    rescale_func = np.poly1d(self.coefficients)
                    self.accumulated_rel_l1_distance_even += rescale_func(((modulated_inp-self.previous_e0_even).abs().mean() / self.previous_e0_even.abs().mean()).cpu().item())
                    if self.accumulated_rel_l1_distance_even < self.teacache_thresh:
                        should_calc_even = False
                    else:
                        should_calc_even = True
                        self.accumulated_rel_l1_distance_even = 0
                self.previous_e0_even = modulated_inp.clone()

            else: # odd -> unconditon
                self.is_even = False
                if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc_odd = True
                    self.accumulated_rel_l1_distance_odd = 0
                else: 
                    rescale_func = np.poly1d(self.coefficients)
                    self.accumulated_rel_l1_distance_odd += rescale_func(((modulated_inp-self.previous_e0_odd).abs().mean() / self.previous_e0_odd.abs().mean()).cpu().item())
                    if self.accumulated_rel_l1_distance_odd < self.teacache_thresh:
                        should_calc_odd = False
                    else:
                        should_calc_odd = True
                        self.accumulated_rel_l1_distance_odd = 0
                self.previous_e0_odd = modulated_inp.clone()

        if self.enable_teacache: 
            if self.is_even:
                if not should_calc_even:
                    x += self.previous_residual_even
                else:
                    ori_x = x.clone()
                    for block in self.blocks:
                        x = block(x, **kwargs)
                    self.previous_residual_even = x - ori_x
            else:
                if not should_calc_odd:
                    x += self.previous_residual_odd
                else:
                    ori_x = x.clone()
                    for block in self.blocks:
                        x = block(x, **kwargs)
                    self.previous_residual_odd = x - ori_x

            self.cnt += 1
            if self.cnt >= self.num_steps:
                self.cnt = 0
        else:
            # ===== 第6步：标准Transformer处理 =====
            # 逐层通过32个WanAttentionBlock进行深度特征变换
            # 每一层都进行：自注意力 -> 交叉注意力 -> 前馈网络
            for block in self.blocks:
                x = block(x, **kwargs)  # [batch, seq_len, dim] -> [batch, seq_len, dim]

        # ===== 第7步：输出头处理 =====
        # 将最终特征转换为噪声预测
        x = self.head(x, e)  # [batch, seq_len, dim] -> [batch, seq_len, out_dim*patch_volume]

        # ===== 第8步：unpatchify恢复视频格式 =====
        # 将token序列重新组织为3D视频张量
        x = self.unpatchify(x, grid_sizes)  # -> [batch, out_dim, F, H/8, W/8]

        # 返回float32格式的噪声预测，用于后续扩散去噪步骤
        return x.float()

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        bs = x.shape[0]
        x = x.view(bs, *grid_sizes, *self.patch_size, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        x = x.reshape(bs, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])

        return x

    def set_ar_attention(self, causal_block_size):
        self.num_frame_per_block = causal_block_size
        self.flag_causal_attention = True
        for block in self.blocks:
            block.set_ar_attention()

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        if self.inject_sample_info:
            nn.init.normal_(self.fps_embedding.weight, std=0.02)

            for m in self.fps_projection.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)

            nn.init.zeros_(self.fps_projection[-1].weight)
            nn.init.zeros_(self.fps_projection[-1].bias)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
