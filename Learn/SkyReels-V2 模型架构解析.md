# SkyReels-V2 模型架构解析

## 🏗️ 整体架构概览

SkyReels-V2采用**多组件协同架构**，包含4个核心模块：

```
SkyReels-V2 架构
├── 🧠 WanModel (核心Transformer) - 14B/1.3B参数
├── 🎬 WanVAE (3D视频编解码器) 
├── 📝 T5TextEncoder (文本理解)
└── 👁️ CLIP (视觉理解，I2V模式)
```

## 🧠 核心组件1: WanModel (Transformer主干)

这是SkyReels-V2的**大脑**，负责实际的视频生成：

```python
class WanModel:
    def __init__(self):
        # 架构参数 (14B模型配置)
        self.dim = 2048          # 隐藏维度  
        self.ffn_dim = 8192      # FFN中间层维度
        self.num_heads = 16      # 注意力头数
        self.num_layers = 32     # Transformer层数
        self.in_dim = 16         # 输入通道 (VAE潜在空间)
        self.out_dim = 16        # 输出通道
        self.patch_size = (1,2,2) # 3D分块大小 (时间,高,宽)
        
        # 关键组件
        self.patch_embedding    # 3D卷积：视频→Token序列
        self.text_embedding     # 文本特征投影
        self.time_embedding     # 时间步嵌入
        self.blocks            # 32个Transformer Block
        self.head              # 输出头
```

**核心创新点：**
- **3D Patch Embedding**: 将视频分块为时空token
- **RoPE位置编码**: 3维旋转位置编码(时间+空间)  
- **混合注意力**: 自注意力+文本交叉注意力
- **因果掩码**: 支持Diffusion Forcing的灵活掩码

## 🎬 核心组件2: WanVAE (3D视频VAE)

负责**视频↔潜在空间**的转换：

```python
class WanVAE:
    def __init__(self):
        # 编码器结构
        self.encoder = Encoder3d(
            dim_mult=[1, 2, 4, 4],        # 通道倍增序列
            temperal_downsample=[T,T,F],  # 时间下采样模式
            num_res_blocks=2              # 残差块数量
        )
        
        # 解码器结构  
        self.decoder = Decoder3d(...)
        
    # 压缩率设计
    # 输入: [B, 3, T, H, W]
    # 输出: [B, 16, T/4, H/8, W/8]  ← 时间4倍,空间64倍压缩
```

**设计特点：**
- **3D因果卷积**: 保持时间因果性  
- **渐进下采样**: 时间4倍、空间64倍压缩
- **分块处理**: 1+4+4+4...帧方式处理，节省显存

## 📝 核心组件3: T5文本编码器

```python 
class T5TextEncoder:
    # 基于T5-XXL模型
    # 文本长度: 512 tokens
    # 输出维度: 4096 → 2048 (投影到Transformer维度)
```

## 👁️ 核心组件4: CLIP视觉编码器 (I2V专用)

```python
class XLMRobertaCLIP:
    def __init__(self):
        self.vision_layers = 32      # 视觉Transformer层数
        self.vision_dim = 1280       # 视觉特征维度
        self.text_layers = 24        # 文本层数
        # 输出: 1280维图像特征 → 2048维投影
```

## 🔧 架构设计亮点

### 1. 多尺度注意力机制

```python
class WanAttentionBlock:
    def __init__(self):
        # 自注意力: 处理时空依赖  
        self.self_attn = FlashAttention(
            dim=2048, 
            num_heads=16,
            qk_norm=True  # Query/Key归一化
        )
        
        # 交叉注意力: 文本条件引导
        self.cross_attn = CrossAttention(
            query_dim=2048,
            cross_dim=4096,  # T5文本特征
            heads=16
        )
        
        # FFN: Swish激活的前馈网络
        self.ffn = FeedForward(2048, 8192)
```

### 2. 3D RoPE位置编码

```python
def rope_apply(x, grid_sizes, freqs):
    f, h, w = grid_sizes  # 时间,高,宽维度
    
    # 分别为3个维度计算旋转频率
    freqs_t = freqs[0][:f]  # 时间维度
    freqs_h = freqs[1][:h]  # 高度维度  
    freqs_w = freqs[2][:w]  # 宽度维度
    
    # 组合3D位置编码
    freqs_3d = torch.cat([freqs_t, freqs_h, freqs_w], dim=-1)
    return x * freqs_3d  # 应用旋转变换
```

### 3. 模式自适应设计

```python
# T2V模式: 纯文本条件
if model_type == "t2v":
    cross_attn_type = "t2v_cross_attn"  # 只有文本交叉注意力
    
# I2V模式: 文本+图像条件  
elif model_type == "i2v":
    cross_attn_type = "i2v_cross_attn"  # 文本+图像交叉注意力
    self.img_emb = MLPProj(1280, 2048)  # CLIP图像投影
```

## 📈 参数规模对比

| 模型版本 | Transformer参数 | VAE参数 | 总参数 | 显存需求 |
|---------|---------------|---------|--------|----------|
| **1.3B** | ~1.3B | ~100M | ~1.4B | ~14.7GB |
| **14B** | ~14B | ~100M | ~14.1B | ~51GB |

## 🔄 前向传播流程

```python
def forward(self, x, t, context, fps=None):
    # 1. 3D Patch Embedding  
    x = self.patch_embedding(x)  # [B,16,T,H,W] → [B,seq_len,2048]
    
    # 2. 时间嵌入
    t_emb = self.time_embedding(sinusoidal_embedding(t))
    
    # 3. 条件嵌入
    context = self.text_embedding(context)  # T5文本特征
    
    # 4. 32层Transformer处理
    for block in self.blocks:
        x = block(
            x,                    # 视频特征
            context=context,      # 文本条件  
            timestep_emb=t_emb,   # 时间信息
            clip_fea=clip_fea     # 图像条件(I2V)
        )
    
    # 5. 输出投影
    x = self.head(x, t_emb)  # 预测噪声
    return self.unpatchify(x, grid_sizes)  # 重构为3D
```

## 💡 Diffusion Forcing 核心创新

**传统扩散 vs Diffusion Forcing**

```python
# 传统扩散：所有帧共享相同噪声级别
# t: [999, 999, 999, 999, ...]  ← 同步扩散

# Diffusion Forcing：每个帧独立噪声级别  
# t: [999, 800, 600, 400, 200, 0, ...]  ← 异步扩散
```

**核心算法逻辑：**
```python
def generate_timestep_matrix(self, num_frames, step_template, ar_step=5):
    # 为每帧分配独立时间步
    while not_all_frames_denoised:
        new_row = torch.zeros(num_frames_block)
        for i in range(num_frames_block):
            if i == 0 or prev_frame_completed:
                new_row[i] = prev_row[i] + 1  # 前进一步
            else:
                new_row[i] = new_row[i-1] - ar_step  # 自回归延迟
        # 更新掩码，只更新需要的帧
        update_mask = (new_row != prev_row) & (new_row != completed)
```

**关键原理：**
- 🎯 **部分掩码机制**: 噪声=0为完全未掩码，完全噪声=完全掩码
- 🔄 **条件引导**: 使用已去噪的帧指导未去噪帧的恢复
- ⚡ **无限扩展**: 基于最后几帧自回归生成新段落

## 🚀 长视频生成机制

当需要生成超长视频时，使用特殊的分段策略：

```
长视频生成 (30秒/737帧)
├── 第1段: 生成1-97帧
├── 第2段: 用80-97帧作为条件，生成98-177帧  
├── 第3段: 用160-177帧作为条件，生成178-257帧
├── ...
└── 第6段: 生成最后80帧，总计737帧

每段内部使用Diffusion Forcing:
├── 帧1: 时间步999 (最多噪声)
├── 帧2: 时间步800  
├── 帧3: 时间步600
├── ...
└── 帧97: 时间步0 (无噪声)
```

**分段滑窗生成策略：**

```python
def long_video_generation(self, base_frames=97, total_frames=737, overlap=17):
    # 计算迭代次数
    n_iter = 1 + (total_frames - base_frames) // (base_frames - overlap) 
    # 例: 737帧 = 6次迭代，每次重叠17帧
    
    output_video = None
    for i in range(n_iter):
        if output_video is not None:
            # 提取重叠区域作为条件
            prefix_video = output_video[:, -overlap:] 
            prefix_latent = self.vae.encode(prefix_video)  # 编码为潜在
            
        # 当前段落生成 
        current_segment = self.diffusion_forcing_step(
            prompt=prompt,
            prefix=prefix_latent,  # 条件帧
            num_frames=base_frames,
            ar_step=5  # 自回归步数
        )
        
        # 拼接结果(去除重叠)
        if output_video is None:
            output_video = current_segment
        else:
            output_video = torch.cat([
                output_video, 
                current_segment[:, overlap:]  # 跳过重叠部分
            ], dim=1)
    
    return output_video  # 最终长视频
```

## 🎯 关键技术特性

1. **🔀 混合精度训练**: BF16主体 + FP32 VAE
2. **⚡ Flash Attention**: 高效长序列注意力计算  
3. **🧮 Gradient Checkpointing**: 节省显存的反向传播
4. **🔧 动态编译**: PyTorch 2.0 compile加速
5. **🎭 灵活掩码**: 支持同步/异步Diffusion Forcing
6. **💾 TeaCache**: 缓存中间特征，3x推理加速
7. **🚀 xDiT USP**: 多GPU序列并行，线性扩展
8. **🎛️ Memory Offload**: CPU/GPU显存协调管理

## 📊 性能优势

### 计算效率优化
- **TeaCache**: 缓存中间特征，3x推理加速
- **xDiT USP**: 多GPU序列并行，线性扩展
- **Memory Offload**: CPU/GPU显存协调管理

### 质量控制机制
- **SkyCaptioner-V1**: 专业电影语法标注
- **分层注意力**: 时间+空间分离建模  
- **多尺度训练**: 540P→720P渐进提升

### 条件控制精度
- **文本引导**: T5编码器深度理解
- **图像引导**: CLIP视觉特征融合
- **帧率自适应**: FPS嵌入动态调整

## 🔬 核心数学公式

**Diffusion Forcing核心公式：**
```
x_{t-1} = α_t * x_t + σ_t * ε_θ(x_t, t, c)
```
其中每帧的t可以不同，实现异步去噪

**长视频生成递推：**
```  
V_n = DF(prompt, V_{n-1}[-overlap:], frames=base_length)
V_final = Concat(V_1, V_2[overlap:], ..., V_n[overlap:])
```

这种设计使SkyReels-V2成为首个真正意义上的**无限长度视频生成模型**，同时在质量、一致性和效率方面都达到了SOTA水平。其核心创新在于将传统的同步扩散转变为**异步条件扩散**，结合专业的**电影语法理解**和**多阶段优化策略**，实现了从短视频到长视频、从概念理解到视觉呈现的全链路突破。

## 📚 多阶段训练策略

**完整训练流水线：**

```
阶段1: 基础预训练
├── 数据: ~2M高质量平衡视频
├── 模型: Transformer + VAE联合训练  
├── 目标: 基础视频理解能力
└── 时长: 24fps视频数据，97帧输入

阶段2: 强化学习优化
├── 目标: 运动质量提升
├── 方法: Direct Preference Optimization (DPO)
├── 数据: 运动质量配对样本
└── 重点: 大变形运动、物理一致性

阶段3: Diffusion Forcing训练  
├── 目标: 无限长度生成能力
├── 方法: 变长噪声级别训练
├── 架构: AutoRegressive Diffusion Forcing
└── 特点: 每帧独立时间步调度

阶段4: 高质量SFT 
├── 540P SFT: 概念平衡训练
├── 720P SFT: 高分辨率微调
├── 数据: 手工筛选高质量样本  
└── 目标: 视觉质量最终提升
```

---

*创建时间：2025年1月*  
*项目：SkyReels-V2*  
*状态：技术解析完成*
