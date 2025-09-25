# WanVAE编码器核心算法详解

## 📋 概述

本文档详细分析SkyReels-V2中WanVAE编码器的核心算法实现，按照数据流顺序梳理每个处理步骤，并提供精确的代码位置引用。

## 🗂️ 核心文件结构

```
skyreels_v2_infer/modules/vae.py
├── CausalConv3d (L17-L35)          # 3D因果卷积
├── RMS_norm (L38-L57)              # RMS归一化
├── Resample (L61-L126)             # 上下采样模块
├── ResidualBlock (L163-L197)       # 残差块
├── AttentionBlock (L200-L236)      # 注意力块
├── Encoder3d (L239-L334)           # 编码器主体
├── WanVAE_ (L444-L541)            # VAE包装类
└── WanVAE (L571-L639)             # 高级接口类
```

## 🔧 第1步：时间分块预处理

### 代码位置: `skyreels_v2_infer/modules/vae.py:478-501`

```python
def encode(self, x, scale):
    self.clear_cache()
    ## cache
    t = x.shape[2]                    # 获取总帧数
    iter_ = 1 + (t - 1) // 4         # 计算迭代次数: 1 + 96//4 = 25次
    
    ## 对encode输入的x，按时间拆分为1、4、4、4....
    for i in range(iter_):
        self._enc_conv_idx = [0]
        if i == 0:
            # 第1块：仅处理第1帧 [B, 3, 1, H, W]
            out = self.encoder(x[:, :, :1, :, :], 
                             feat_cache=self._enc_feat_map, 
                             feat_idx=self._enc_conv_idx)
        else:
            # 后续块：每次4帧 [B, 3, 4, H, W]
            out_ = self.encoder(
                x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
            )
            out = torch.cat([out, out_], 2)  # 时间维度拼接
```

**核心原理:**
- 🎯 **第1帧特殊处理**: 建立时间基准，确保因果性
- 🔄 **4帧一组**: 平衡计算效率与时间建模能力  
- 💾 **缓存机制**: `feat_cache`保存中间特征，优化显存使用

## ⚡ 第2步：3D因果卷积初始化

### 代码位置: `skyreels_v2_infer/modules/vae.py:17-35`

```python
class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 设置因果填充: (W_pad, W_pad, H_pad, H_pad, T_pad_left, T_pad_right)
        self._padding = (self.padding[2], self.padding[2], 
                        self.padding[1], self.padding[1], 
                        2 * self.padding[0], 0)  # 时间维度只向前填充
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)      # 时间维度拼接缓存
            padding[4] -= cache_x.shape[2]         # 调整填充大小
        x = F.pad(x, padding)                       # 应用因果填充
        return super().forward(x)                   # 执行3D卷积
```

### 编码器初始化: `skyreels_v2_infer/modules/vae.py:262-263`

```python
# init block  
self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)  # 3→96通道
```

**数学变换:**
```
输入: [B, 3, T, H, W]
Conv3d: kernel=3×3×3, stride=1×1×1, padding=因果填充
输出: [B, 96, T, H, W]
```

## 🔽 第3步：多层渐进下采样

### 下采样架构设计: `skyreels_v2_infer/modules/vae.py:258-280`

```python
# dimensions
dims = [dim * u for u in [1] + dim_mult]  # [96, 192, 384, 384]
scale = 1.0

# downsample blocks
downsamples = []
for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
    # residual (+attention) blocks
    for _ in range(num_res_blocks):  # 默认2个残差块
        downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
        if scale in attn_scales:  # 在指定尺度添加注意力
            downsamples.append(AttentionBlock(out_dim))
        in_dim = out_dim

    # downsample block
    if i != len(dim_mult) - 1:
        # 根据配置选择下采样模式
        mode = "downsample3d" if temperal_downsample[i] else "downsample2d" 
        downsamples.append(Resample(out_dim, mode=mode))
        scale /= 2.0
self.downsamples = nn.Sequential(*downsamples)
```

### 残差块实现: `skyreels_v2_infer/modules/vae.py:163-197`

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 主路径：双卷积结构
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),           # RMS归一化
            nn.SiLU(),                               # Swish激活函数
            CausalConv3d(in_dim, out_dim, 3, padding=1),  # 第1个3D卷积
            RMS_norm(out_dim, images=False),          # 再次归一化
            nn.SiLU(),                               # 再次激活
            nn.Dropout(dropout),                     # Dropout正则化
            CausalConv3d(out_dim, out_dim, 3, padding=1), # 第2个3D卷积
        )
        # 跳跃连接：维度匹配
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)  # 跳跃连接处理
        
        # 逐层处理主路径，支持缓存优化
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                # 缓存最后2帧用于下次计算
                cache_x = x[:, :, -CACHE_T:, :, :].clone() 
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), 
                        cache_x
                    ], dim=2)
                x = layer(x, feat_cache[idx])  # 带缓存的卷积
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h  # 残差相加
```

### 配置参数详解

从 `skyreels_v2_infer/modules/vae.py:549-557`:
```python
cfg = dict(
    dim=96,                              # 基础通道数
    z_dim=z_dim,                        # 潜在空间维度 (通常16)
    dim_mult=[1, 2, 4, 4],             # 通道倍增序列
    num_res_blocks=2,                   # 每层残差块数量
    attn_scales=[],                     # 注意力层位置
    temperal_downsample=[False, True, True],  # 时间下采样策略
    dropout=0.0,                        # Dropout概率
)
```

**维度变化追踪:**

| 层级 | 输入维度 | 残差处理后 | 下采样后 | 下采样类型 |
|------|----------|------------|----------|------------|
| **Init** | `[B,3,T,H,W]` | `[B,96,T,H,W]` | - | CausalConv3d |
| **Level-0** | `[B,96,T,H,W]` | `[B,192,T,H,W]` | `[B,192,T,H/2,W/2]` | downsample2d |
| **Level-1** | `[B,192,T,H/2,W/2]` | `[B,384,T,H/2,W/2]` | `[B,384,T/2,H/4,W/4]` | downsample3d |
| **Level-2** | `[B,384,T/2,H/4,W/4]` | `[B,384,T/2,H/4,W/4]` | `[B,384,T/4,H/8,W/8]` | downsample3d |

## 🌊 第4步：中间特征处理

### 代码位置: `skyreels_v2_infer/modules/vae.py:282-285`

```python
# middle blocks
self.middle = nn.Sequential(
    ResidualBlock(out_dim, out_dim, dropout),  # 第1个残差块
    AttentionBlock(out_dim),                   # 自注意力机制
    ResidualBlock(out_dim, out_dim, dropout)   # 第2个残差块  
)
```

### 注意力块实现: `skyreels_v2_infer/modules/vae.py:200-236`

```python
class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.to_q = nn.Conv2d(dim, dim, 1)      # Query投影
        self.to_k = nn.Conv2d(dim, dim, 1)      # Key投影  
        self.to_v = nn.Conv2d(dim, dim, 1)      # Value投影
        self.to_out = nn.Conv2d(dim, dim, 1)    # 输出投影

    def forward(self, x):
        # 输入: [B, C, T, H, W]
        identity = x
        b, c, t, h, w = x.shape

        # 重排为2D格式进行注意力计算
        x = rearrange(x, "b c t h w -> (b t) c h w")

        # 注意力机制
        q = self.to_q(x)  # Query: [(BT), C, H, W]
        k = self.to_k(x)  # Key: [(BT), C, H, W]  
        v = self.to_v(x)  # Value: [(BT), C, H, W]

        # 计算注意力分数 (简化的空间注意力)
        q = rearrange(q, "bt c h w -> bt (h w) c")
        k = rearrange(k, "bt c h w -> bt c (h w)")
        v = rearrange(v, "bt c h w -> bt (h w) c")

        # 缩放点积注意力
        scale = c ** -0.5
        attn = torch.softmax(q @ k * scale, dim=-1)  # [(BT), HW, HW]
        out = attn @ v  # [(BT), HW, C]

        # 重构输出形状
        out = rearrange(out, "bt (h w) c -> (b t) c h w", h=h, w=w)
        out = self.to_out(out)
        
        # 恢复时间维度并添加残差连接
        x = rearrange(out, "(b t) c h w -> b c t h w", t=t)
        return x + identity
```

## 📤 第5步：输出头生成潜在特征

### 输出头定义: `skyreels_v2_infer/modules/vae.py:287-290`

```python
# output blocks
self.head = nn.Sequential(
    RMS_norm(out_dim, images=False),              # 最终归一化
    nn.SiLU(),                                   # Swish激活
    CausalConv3d(out_dim, z_dim, 3, padding=1)   # 384→16通道
)
```

### VAE参数生成: `skyreels_v2_infer/modules/vae.py:495-500`

```python
# 在WanVAE_.encode方法中
mu, log_var = self.conv1(out).chunk(2, dim=1)  # 分离均值和对数方差
if isinstance(scale[0], torch.Tensor):
    # 应用预训练的统计量进行标准化  
    mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
else:
    mu = (mu - scale[0]) * scale[1]
```

### 标准化统计量: `skyreels_v2_infer/modules/vae.py:574-613`

```python
# 预训练的通道级统计量
mean = [
    -0.7571, -0.7089, -0.9113,  0.1075,  # 通道 0-3
    -0.1745,  0.9653, -0.1517,  1.5508,  # 通道 4-7
     0.4134, -0.0715,  0.5517, -0.3632,  # 通道 8-11
    -0.1922, -0.9497,  0.2503, -0.2921,  # 通道 12-15
]
std = [
    2.8184, 1.4541, 2.3275, 2.6558,     # 通道 0-3 标准差
    1.2196, 1.7708, 2.6052, 2.0743,     # 通道 4-7 标准差
    3.2687, 2.1526, 2.8652, 1.5579,     # 通道 8-11 标准差
    1.6382, 1.1253, 2.8251, 1.9160,     # 通道 12-15 标准差
]
self.vae_stride = (4, 8, 8)             # 时空压缩倍率
self.scale = [self.mean, 1.0 / self.std] # 标准化参数
```

## 🏁 完整算法流程总结

### 高级接口调用: `skyreels_v2_infer/modules/vae.py:625-629`

```python
def encode(self, video):
    """
    videos: A list of videos each with shape [C, T, H, W].
    """
    return self.vae.encode(video, self.scale).float()
```

### 核心算法管道

```python
def wan_vae_encoder_complete_pipeline():
    """
    基于代码位置的完整编码流程
    
    文件路径: skyreels_v2_infer/modules/vae.py
    """
    
    # ========== 阶段1: 时间分块 (L478-501) ==========
    # 1+4+4+4...帧分块处理，优化显存使用
    iter_ = 1 + (T - 1) // 4
    chunks = process_temporal_chunks(input_video)
    
    # ========== 阶段2: 初始卷积 (L262-263, L299-303) ==========  
    # RGB → 96通道特征提取
    x = CausalConv3d(3→96, kernel=3×3×3)(input)  # [B,3,T,H,W] → [B,96,T,H,W]
    
    # ========== 阶段3: 下采样编码 (L305-310) ==========
    # Level 0: 96→192, 仅空间下采样
    x = ResidualBlock(96→192) × 2           # [B,96,T,H,W] → [B,192,T,H,W]  
    x = Downsample2D()                      # [B,192,T,H,W] → [B,192,T,H/2,W/2]
    
    # Level 1: 192→384, 时空同时下采样
    x = ResidualBlock(192→384) × 2          # [B,192,T,H/2,W/2] → [B,384,T,H/2,W/2]
    x = Downsample3D()                      # [B,384,T,H/2,W/2] → [B,384,T/2,H/4,W/4]
    
    # Level 2: 384→384, 时空同时下采样  
    x = ResidualBlock(384→384) × 2          # [B,384,T/2,H/4,W/4] → [B,384,T/2,H/4,W/4]
    x = Downsample3D()                      # [B,384,T/2,H/4,W/4] → [B,384,T/4,H/8,W/8]
    
    # ========== 阶段4: 中间处理 (L312-317) ==========
    # 残差-注意力-残差结构
    x = ResidualBlock(384→384)              # 特征提取
    x = AttentionBlock(384)                 # 空间自注意力  
    x = ResidualBlock(384→384)              # 特征融合
    
    # ========== 阶段5: 潜在生成 (L319-334) ==========
    # 生成VAE参数
    x = RMS_norm(x) + SiLU(x) + CausalConv3d(384→32)(x)
    mu, log_var = x.chunk(2, dim=1)        # [B,384,...] → [B,16,...], [B,16,...]
    
    # VAE重参数化 + 标准化
    z = mu + ε * exp(0.5 * log_var)        # 采样潜在变量
    z_normalized = (z - mean) * scale       # 应用预训练统计量
    
    return z_normalized  # [B, 16, T/4, H/8, W/8]
```

## 📊 性能与压缩统计

### 压缩效果分析
```python
# 基于 self.vae_stride = (4, 8, 8) 的压缩设计
原始视频: [B, 3, 97, 720, 1280]    ≈ 267MB (FP32)
潜在特征: [B, 16, 24, 90, 160]      ≈ 22.3MB (FP32)
压缩比: 267MB / 22.3MB = 12:1

# 维度压缩明细
时间压缩: 97帧 → 24帧 (4倍)        # temperal_downsample配置
空间压缩: 720×1280 → 90×160 (64倍)  # 3层2倍下采样: 2³=8, 8×8=64
通道变化: 3 → 16 (5.33倍增加)       # 信息密度提升
```

### 关键创新点

| 技术特性 | 代码实现位置 | 核心作用 |
|----------|--------------|----------|
| **🔗 因果性保证** | `CausalConv3d (L17-35)` | 时间序列因果建模，支持实时推理 |
| **💾 分块优化** | `encode方法 (L478-501)` | 1+4+4处理策略，优化长视频显存 |
| **🎯 渐进压缩** | `下采样配置 (L265-280)` | 多尺度特征提取，平衡质量与效率 |
| **⚡ 注意力增强** | `AttentionBlock (L200-236)` | 空间全局建模，提升特征表达力 |
| **🧮 VAE框架** | `重参数化 (L522-525)` | 连续潜在分布，支持生成建模 |
| **📏 统计标准化** | `预训练统计量 (L574-613)` | 稳定潜在空间分布，加速收敛 |

## 🔗 相关组件接口

### 与Transformer的接口
```python
# 在 skyreels_v2_infer/modules/transformer.py:761-770
def forward(self, x, t, context, clip_fea=None, y=None, fps=None):
    """
    x: 输入视频张量 [batch, C_in=16, F, H, W] (VAE潜在特征) ← WanVAE输出
    """
```

### 模型初始化示例
```python  
# 基于 skyreels_v2_infer/modules/vae.py:571-623
vae = WanVAE(
    vae_pth="cache/vae_step_411000.pth",    # 预训练权重路径
    z_dim=16                                # 潜在空间维度
)

# 编码过程
video_tensor = torch.randn(1, 3, 97, 720, 1280)  # 输入视频
latent_features = vae.encode(video_tensor)         # [1, 16, 24, 90, 160]
```

这个设计使WanVAE成为SkyReels-V2架构中的**核心压缩模块**，通过**因果3D卷积**、**分块处理**和**多尺度下采样**的完美结合，实现了高效的视频到潜在空间的映射，为后续的Transformer生成奠定了坚实基础。

---

*文档创建时间：2025年1月*  
*代码版本：SkyReels-V2*  
*分析完成状态：✅ 完整*
