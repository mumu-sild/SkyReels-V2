# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SkyReels-V2 WanVAE 3D视频自编码器实现
# 功能：将视频从像素空间压缩到潜在空间，并支持重建
# 核心特性：因果3D卷积、时间分块处理、多尺度下采样

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


__all__ = [
    "WanVAE",
]

# 缓存帧数：用于时间因果性保持，缓存最后2帧用于下次计算
CACHE_T = 2

# 
class CausalConv3d(nn.Conv3d):
    """
    因果3D卷积：确保时间维度的因果性，即当前帧只依赖于过去帧
    
    核心原理：
    - 在时间维度上只向前填充，不向后填充
    - 保证视频生成的时间一致性和因果性
    - 支持缓存机制优化长视频处理
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.padding：Conv3d 自带的“内部填充参数”（3 个数，顺序是 (T, H, W)）
        # 设置因果填充：(宽度左, 宽度右, 高度上, 高度下, 时间前, 时间后)
        # 时间维度：只向前填充(2*padding)，向后填充为0，确保因果性
        self._padding = (
            self.padding[2], self.padding[2],   # W_left, W_right
            self.padding[1], self.padding[1],   # H_top, H_bottom
            2 * self.padding[0], 0              # T_front, T_back (因果：只前不后)
        )

        self.padding = (0, 0, 0)  # 重置原始填充，使用自定义填充

    def forward(self, x, cache_x=None):
        """
        前向传播
        
        Args:
            x: 输入张量 [B, C, T, H, W]
            cache_x: 缓存的历史帧，用于保持时间连续性
            
        Returns:
            卷积后的特征张量
        """
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            # 如果有缓存帧，在时间维度前端拼接
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)  # 时间维度拼接
            padding[4] -= cache_x.shape[2]     # 调整填充大小
        
        # 应用因果填充
        x = F.pad(x, padding)
        
        # 执行标准3D卷积
        return super().forward(x)


class RMS_norm(nn.Module):
    """
    RMS归一化：Root Mean Square Normalization
    
    相比LayerNorm更稳定，计算效率更高
    公式: RMS(x) = x / sqrt(mean(x^2)) * scale * gamma + bias
    """
    
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        """
        Args:
            dim: 特征维度
            channel_first: 通道维度是否在前 (BCHW vs BHWC)
            images: 是否为图像数据 (影响广播维度)
            bias: 是否使用偏置项
        """
        super().__init__()
        # 根据数据格式设置广播维度
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5  # 缩放因子：sqrt(dim)
        self.gamma = nn.Parameter(torch.ones(shape))   # 可学习的缩放参数
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0  # 可学习的偏置参数

    def forward(self, x):
        """
        前向传播：RMS归一化
        
        Args:
            x: 输入张量
            
        Returns:
            归一化后的张量
        """
        # 沿通道维度进行L2归一化，然后乘以缩放因子和可学习参数
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):
    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    """
    重采样模块：支持2D/3D上采样和下采样
    
    关键特性：
    - 2D模式：仅在空间维度(H,W)进行采样
    - 3D模式：同时在时间和空间维度(T,H,W)进行采样
    - 支持缓存机制，优化长视频处理
    """
    
    def __init__(self, dim, mode):
        """
        Args:
            dim: 输入通道数
            mode: 重采样模式 {'none', 'upsample2d', 'upsample3d', 'downsample2d', 'downsample3d'}
        """
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        super().__init__()
        self.dim = dim
        self.mode = mode

        # 根据模式构建不同的重采样层
        if mode == "upsample2d":
            # 2D上采样：仅空间维度2倍放大
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), 
                nn.Conv2d(dim, dim // 2, 3, padding=1)
            )
        elif mode == "upsample3d":
            # 3D上采样：空间维度2倍放大 + 时间维度处理
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), 
                nn.Conv2d(dim, dim // 2, 3, padding=1)
            )
            # 时间卷积：处理时间维度上采样
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == "downsample2d":
            # 2D下采样：仅空间维度2倍缩小
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),      # 零填充确保尺寸匹配
                nn.Conv2d(dim, dim, 3, stride=(2, 2))  # 步长2的卷积实现下采样
            )
        elif mode == "downsample3d":
            # 3D下采样：时间和空间维度同时2倍缩小
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),      # 空间维度零填充
                nn.Conv2d(dim, dim, 3, stride=(2, 2))  # 空间维度下采样
            )
            # 时间卷积：时间维度下采样
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            # 无操作模式
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        # cache last frame of last two chunk
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                        )
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -1:, :, :].clone()
                    # if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        # conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  # * 0.5
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        # init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        conv_weight[: c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2 :, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):
    """
    残差块：ResNet风格的残差连接块
    
    结构：输入 -> [RMS_norm -> SiLU -> Conv3D] x2 -> 输出
           |                                    |
           +---------> shortcut连接 -----------+
           
    特性：
    - 使用RMS归一化替代BatchNorm
    - SiLU(Swish)激活函数，平滑且可微
    - 3D因果卷积保持时间因果性
    - 支持通道数变换的跳跃连接
    """
    
    def __init__(self, in_dim, out_dim, dropout=0.0):
        """
        Args:
            in_dim: 输入通道数
            out_dim: 输出通道数
            dropout: Dropout概率
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 主路径：双卷积结构
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),           # RMS归一化
            nn.SiLU(),                               # Swish激活函数
            CausalConv3d(in_dim, out_dim, 3, padding=1),  # 第1个3D因果卷积
            RMS_norm(out_dim, images=False),          # 再次归一化
            nn.SiLU(),                               # 再次激活
            nn.Dropout(dropout),                     # Dropout正则化
            CausalConv3d(out_dim, out_dim, 3, padding=1), # 第2个3D因果卷积
        )
        
        # 跳跃连接：处理通道数不匹配的情况
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    自注意力块：因果性单头自注意力机制
    
    功能：
    - 在空间维度(H,W)上进行自注意力计算
    - 每个时间步独立处理，保持时间因果性
    - 增强空间全局建模能力
    
    结构：输入 -> RMS_norm -> QKV -> ScaledDotProductAttention -> 输出投影 -> 残差连接
    """

    def __init__(self, dim):
        """
        Args:
            dim: 特征维度
        """
        super().__init__()
        self.dim = dim

        # 网络层定义
        self.norm = RMS_norm(dim)                    # RMS归一化
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)     # 1x1卷积生成Q,K,V
        self.proj = nn.Conv2d(dim, dim, 1)           # 输出投影层

        # 零初始化最后一层权重，确保训练初期的稳定性
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        """
        前向传播：自注意力计算
        
        Args:
            x: 输入张量 [B, C, T, H, W]
            
        Returns:
            注意力处理后的张量 [B, C, T, H, W]
        """
        identity = x  # 保存输入用于残差连接
        b, c, t, h, w = x.size()
        
        # 重排维度：将时间维度合并到批次维度，每个时间步独立处理
        x = rearrange(x, "b c t h w -> (b t) c h w")  # [BT, C, H, W]
        x = self.norm(x)  # RMS归一化
        
        # 计算Query, Key, Value
        qkv = self.to_qkv(x)  # [BT, 3C, H, W]
        q, k, v = qkv.reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        # q,k,v: [BT, 1, HW, C]

        # 缩放点积注意力计算
        x = F.scaled_dot_product_attention(
            q,  # Query: [BT, 1, HW, C]
            k,  # Key: [BT, 1, HW, C]
            v,  # Value: [BT, 1, HW, C]
        )   # 输出: [BT, 1, HW, C]
        
        # 重塑输出形状
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)  # [BT, C, H, W]

        # 输出投影
        x = self.proj(x)  # [BT, C, H, W]
        
        # 恢复时间维度
        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)  # [B, C, T, H, W]
        
        # 残差连接
        return x + identity


class Encoder3d(nn.Module):
    """
    3D编码器：将视频从像素空间编码到潜在空间
    
    架构：
    1. 初始3D卷积：RGB -> 特征空间
    2. 多层下采样：渐进压缩时空分辨率
    3. 中间处理：残差块 + 注意力 + 残差块
    4. 输出头：生成潜在表示
    
    压缩策略：
    - 时间维度：根据temperal_downsample配置压缩
    - 空间维度：每层2倍下采样
    - 通道维度：按dim_mult倍增特征数
    """
    
    def __init__(
        self,
        dim=128,                              # 基础特征维度
        z_dim=4,                             # 潜在空间维度
        dim_mult=[1, 2, 4, 4],              # 各层通道倍数
        num_res_blocks=2,                    # 每层残差块数量
        attn_scales=[],                      # 在哪些尺度添加注意力
        temperal_downsample=[True, True, False],  # 各层是否进行时间下采样
        dropout=0.0,                         # Dropout概率
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # 计算各层特征维度：[96, 192, 384, 384] (假设dim=96)
        # [1] + dim_mult 把单元素列表 [1] 和 dim_mult 连接起来。
        # 如果 dim_mult = [1, 2, 4, 4]，那么结果是 [1, 1, 2, 4, 4]。
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0  # 用于追踪下采样尺度

        # 初始卷积块：RGB(3通道) -> 特征空间(dim[0]=96通道),kernel=3×3×3, 填充padding=1
        # 通道数变为 dims[0]，保持因果性且不改变时间/空间长度
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # 多层下采样块构建
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # 每层包含num_res_blocks个残差块
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                # 在指定尺度添加注意力机制
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim  # 更新输入维度

            # 添加下采样层（除了最后一层）
            if i != len(dim_mult) - 1:
                # 根据配置选择2D或3D下采样
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0  # 更新尺度追踪
        self.downsamples = nn.Sequential(*downsamples)

        # 中间处理块：残差 -> 注意力 -> 残差
        # 在最深层进行全局特征整合
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),  # 特征提取
            AttentionBlock(out_dim),                   # 空间全局建模
            ResidualBlock(out_dim, out_dim, dropout)   # 特征融合
        )

        # 输出头：生成潜在表示
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),           # 最终归一化
            nn.SiLU(),                                # Swish激活
            CausalConv3d(out_dim, z_dim, 3, padding=1) # 映射到潜在空间维度
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]

            # 缓存最后2帧特征图
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            
            # 执行3D卷积
            x = self.conv1(x, feat_cache[idx])

            # 更新缓存
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):
    """
    3D解码器：将潜在空间特征解码回视频像素空间
    
    架构：
    1. 初始3D卷积：潜在特征 -> 特征空间
    2. 中间处理：残差块 + 注意力 + 残差块
    3. 多层上采样：逐步恢复时空分辨率
    4. 输出头：生成RGB视频
    
    上采样策略：
    - 时间维度：根据temperal_upsample配置恢复
    - 空间维度：每层2倍上采样
    - 通道维度：按dim_mult逐步减少特征数
    
    注意：这是编码器的镜像结构，确保信息完整恢复
    """
    
    def __init__(
        self,
        dim=128,                              # 基础特征维度
        z_dim=4,                             # 潜在空间维度
        dim_mult=[1, 2, 4, 4],              # 各层通道倍数（与编码器相同）
        num_res_blocks=2,                    # 每层残差块数量
        attn_scales=[],                      # 在哪些尺度添加注意力
        temperal_upsample=[False, True, True], # 各层是否进行时间上采样
        dropout=0.0,                         # Dropout概率
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # 计算各层特征维度：[384, 384, 192, 96]（与编码器镜像）
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)  # 初始尺度

        # 初始卷积块：潜在空间 -> 特征空间
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # 中间处理块：与编码器相同的结构
        # 在最深层进行全局特征整合
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),  # 特征提取
            AttentionBlock(dims[0]),                   # 空间全局建模
            ResidualBlock(dims[0], dims[0], dropout)   # 特征融合
        )

        # 多层上采样块构建（编码器的逆过程）
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # 调整输入维度（由于上采样操作的维度变化）
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2  # 上采样后通道数会减半
                
            # 每层包含num_res_blocks+1个残差块
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                # 在指定尺度添加注意力机制
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim  # 更新输入维度

            # 添加上采样层（除了最后一层）
            if i != len(dim_mult) - 1:
                # 根据配置选择2D或3D上采样
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0  # 更新尺度追踪
        self.upsamples = nn.Sequential(*upsamples)

        # 输出头：生成RGB视频
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),           # 最终归一化
            nn.SiLU(),                                # Swish激活
            CausalConv3d(out_dim, 3, 3, padding=1)    # 映射到RGB 3通道
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(
            dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, self.temperal_downsample, dropout
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks, attn_scales, self.temperal_upsample, dropout)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale):
        """
        编码方法：将视频编码到潜在空间
        
        采用分块处理策略：1+4+4+4...帧
        - 第1帧单独处理：建立时间基准
        - 后续4帧一组：平衡效率与因果性
        - 使用缓存机制：优化长视频显存使用
        
        Args:
            x: 输入视频 [B, 3, T, H, W]
            scale: 标准化参数 [mean, std]
            
        Returns:
            mu: 潜在表示的均值 [B, z_dim, T/4, H/8, W/8]
        """
        self.clear_cache()  # 清空缓存
        
        # 计算时间分块策略
        t = x.shape[2]               # 获取时间维度长度
        iter_ = 1 + (t - 1) // 4     # 迭代次数：1(首帧) + ceil((T-1)/4)
        
        # 分块编码循环
        for i in range(iter_):
            self._enc_conv_idx = [0]  # 重置卷积索引
            
            if i == 0:
                # 第1块：仅处理第1帧，建立时间基准
                out = self.encoder(
                    x[:, :, :1, :, :],                    # [B, 3, 1, H, W]
                    feat_cache=self._enc_feat_map,        # 特征缓存
                    feat_idx=self._enc_conv_idx           # 索引追踪
                )
            else:
                # 后续块：每次处理4帧
                start_idx = 1 + 4 * (i - 1)
                end_idx = 1 + 4 * i
                out_ = self.encoder(
                    x[:, :, start_idx:end_idx, :, :],     # [B, 3, 4, H, W]
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                # 在时间维度拼接结果
                out = torch.cat([out, out_], dim=2)
        
        # VAE参数生成：生成均值和对数方差
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        
        # 应用预训练的标准化统计量
        if isinstance(scale[0], torch.Tensor):
            # 张量格式：逐通道标准化
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
        else:
            # 标量格式：全局标准化
            mu = (mu - scale[0]) * scale[1]
        
        self.clear_cache()  # 清空缓存
        return mu

    def decode(self, z, scale):
        """
        解码方法：将潜在空间特征解码为视频
        
        采用逐帧处理策略：
        - 对每一帧潜在特征单独解码
        - 使用缓存机制保持时间连续性
        - 在时间维度拼接所有解码帧
        
        Args:
            z: 潜在特征 [B, z_dim, T/4, H/8, W/8]
            scale: 标准化参数 [mean, std]
            
        Returns:
            reconstructed: 重建视频 [B, 3, T, H, W]
        """
        self.clear_cache()  # 清空缓存
        
        # 反标准化：将标准化的潜在特征恢复到原始分布
        if isinstance(scale[0], torch.Tensor):
            # 张量格式：逐通道反标准化
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        else:
            # 标量格式：全局反标准化
            z = z / scale[1] + scale[0]
        
        # 获取时间维度长度并预处理
        iter_ = z.shape[2]  # 潜在空间的时间帧数
        x = self.conv2(z)   # 通道调整
        
        # 逐帧解码循环
        for i in range(iter_):
            self._conv_idx = [0]  # 重置卷积索引
            
            if i == 0:
                # 第1帧：建立时间基准
                out = self.decoder(
                    x[:, :, i : i + 1, :, :],          # 单帧输入 [B, z_dim, 1, H/8, W/8]
                    feat_cache=self._feat_map,         # 特征缓存
                    feat_idx=self._conv_idx            # 索引追踪
                )
            else:
                # 后续帧：利用缓存保持连续性
                out_ = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
                # 在时间维度拼接结果
                out = torch.cat([out, out_], dim=2)
        
        self.clear_cache()  # 清空缓存
        return out  # [B, 3, T, H, W]

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        """
        清空缓存：重置所有缓存相关的状态变量
        
        缓存机制用于优化长视频处理：
        - 保存中间特征图，避免重复计算
        - 维护时间因果性，确保帧间依赖正确
        - 支持分块处理，降低显存需求
        """
        # 解码器缓存初始化
        self._conv_num = count_conv3d(self.decoder)    # 统计解码器中3D卷积层数
        self._conv_idx = [0]                          # 当前卷积层索引
        self._feat_map = [None] * self._conv_num      # 解码器特征缓存数组
        
        # 编码器缓存初始化

        # int 编码器中3D卷积层数 = 5
        self._enc_conv_num = count_conv3d(self.encoder) # 统计编码器中3D卷积层数

        # 格式：[int] 单元素列表（使用列表是为了引用传递）
        # 具体内容：当前正在处理的3D卷积层序号
        self._enc_conv_idx = [0]                       # 当前编码器卷积层索引

        # 初始状态：[None, None, None, None, None]
        # [B, C, 2, H, W] 编码器各层3D卷积输出的最后2帧特征图
        self._enc_feat_map = [None] * self._enc_conv_num # 编码器特征缓存数组


def _video_vae(pretrained_path=None, z_dim=None, device="cpu", **kwargs):
    """
    VAE模型工厂函数：创建并加载预训练的3D视频VAE模型
    
    该函数基于Stable Diffusion的自编码器架构，针对视频数据进行了3D扩展
    
    Args:
        pretrained_path: 预训练权重文件路径
        z_dim: 潜在空间维度（通常为16）
        device: 目标设备（'cpu', 'cuda', etc.）
        **kwargs: 其他模型配置参数
        
    Returns:
        加载了预训练权重的WanVAE_模型实例
    """
    # 默认配置参数
    cfg = dict(
        dim=96,                              # 基础特征维度
        z_dim=z_dim,                        # 潜在空间维度
        dim_mult=[1, 2, 4, 4],             # 各层通道倍增序列
        num_res_blocks=2,                   # 每层残差块数量
        attn_scales=[],                     # 注意力层位置（空列表表示不使用）
        temperal_downsample=[False, True, True], # 时间下采样配置
        dropout=0.0,                        # Dropout概率
    )
    cfg.update(**kwargs)  # 更新用户自定义配置

    # 在meta设备上初始化模型（不占用实际内存）
    with torch.device("meta"):
        model = WanVAE_(**cfg)

    # 加载预训练权重
    logging.info(f"loading {pretrained_path}")
    model.load_state_dict(torch.load(pretrained_path, map_location=device), assign=True)

    return model


class WanVAE:
    """
    WanVAE高级封装类：SkyReels-V2的3D视频自编码器
    
    主要功能：
    1. 视频编码：[B,3,T,H,W] -> [B,16,T/4,H/8,W/8]
    2. 视频解码：[B,16,T/4,H/8,W/8] -> [B,3,T,H,W]
    3. 自动标准化：使用预训练统计量
    4. 设备管理：支持GPU/CPU切换
    
    压缩比例：
    - 时间：4倍压缩 (97帧 -> 24帧)
    - 空间：64倍压缩 (720x1280 -> 90x160)
    - 总体：约12倍压缩
    """
    
    def __init__(self, vae_pth="cache/vae_step_411000.pth", z_dim=16):
        """
        Args:
            vae_pth: VAE预训练权重路径
            z_dim: 潜在空间维度，默认16
        """

        # 预训练的16通道标准化统计量（从大量数据计算得出）
        # 每个通道的均值
        mean = [
            -0.7571, -0.7089, -0.9113,  0.1075,  # 通道 0-3
            -0.1745,  0.9653, -0.1517,  1.5508,  # 通道 4-7
             0.4134, -0.0715,  0.5517, -0.3632,  # 通道 8-11
            -0.1922, -0.9497,  0.2503, -0.2921,  # 通道 12-15
        ]
        # 每个通道的标准差
        std = [
            2.8184, 1.4541, 2.3275, 2.6558,     # 通道 0-3
            1.2196, 1.7708, 2.6052, 2.0743,     # 通道 4-7
            3.2687, 2.1526, 2.8652, 1.5579,     # 通道 8-11
            1.6382, 1.1253, 2.8251, 1.9160,     # 通道 12-15
        ]
        
        # 压缩倍率：(时间, 高度, 宽度)
        self.vae_stride = (4, 8, 8)
        
        # 转换为张量并构建标准化参数
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.scale = [self.mean, 1.0 / self.std]  # [均值, 1/标准差]

        # 初始化VAE模型
        self.vae = (
            _video_vae(
                pretrained_path=vae_pth,  # 预训练权重路径
                z_dim=z_dim,             # 潜在空间维度
            )
            .eval()                      # 设置为评估模式
            .requires_grad_(False)       # 冻结参数，不参与梯度更新
        )

    def encode(self, video):
        """
        视频编码：将视频从像素空间编码到标准化的潜在空间
        
        Args:
            video: 输入视频张量 [B, 3, T, H, W]
                  - 值域：[-1, 1] (标准化后的像素值)
                  - 格式：RGB通道顺序
                  
        Returns:
            latent: 潜在表示 [B, 16, T/4, H/8, W/8]
                   - 已应用标准化，均值0，标准差1
        """
        return self.vae.encode(video, self.scale).float()

    def to(self, *args, **kwargs):
        """
        设备转换：将模型和统计参数转移到指定设备
        
        Args:
            *args, **kwargs: PyTorch的to()方法参数（如device, dtype等）
            
        Returns:
            self: 支持链式调用
        """
        # 转移标准化统计参数到目标设备
        self.mean = self.mean.to(*args, **kwargs)
        self.std = self.std.to(*args, **kwargs)
        self.scale = [self.mean, 1.0 / self.std]  # 重新构建标准化参数
        
        # 转移VAE模型到目标设备
        self.vae = self.vae.to(*args, **kwargs)
        return self

    def decode(self, z):
        """
        视频解码：将潜在表示解码回视频像素空间
        
        Args:
            z: 潜在表示 [B, 16, T/4, H/8, W/8]
               - 标准化的潜在特征
               
        Returns:
            video: 重建视频 [B, 3, T, H, W]
                  - 值域：[-1, 1] (限制在合理范围)
                  - 格式：RGB通道顺序
        """
        return self.vae.decode(z, self.scale).float().clamp_(-1, 1)
