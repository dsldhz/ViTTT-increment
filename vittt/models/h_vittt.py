# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# ViT^3: Unlocking Test-Time Training in Vision
# Modified by Dongchen Han
# --------------------------------------------------------

# 本文件实现“分层（Hierarchical）”ViTTT，整体思路类似 Swin/PVT：
# 1. Stem 用卷积把图像变成较高分辨率的 token；
# 2. 每个 stage 内堆叠多个 TTTBlock；
# 3. stage 之间用 PatchMerging 将高、宽减半并提高通道数；
# 4. 最后对最低分辨率特征做全局池化和分类。

# PyTorch 核心模块和梯度检查点工具。
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

# timm 中的随机深度、二元尺寸转换和截断正态初始化。
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# ViTTT 的核心序列建模模块。
from .ttt_block import TTT


class Mlp(nn.Module):
    """带深度卷积的前馈网络。

    与标准 Transformer MLP 相比，中间加入一个 3×3 深度卷积，
    使逐 token 的通道变换也能感知相邻空间位置。输入/输出均为 ``[B,N,C]``。
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        # 未指定输出/隐藏维度时，分别回退到输入维度。
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # 第一层扩展通道，深度卷积混合空间信息，第二层再投影到输出维度。
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwc = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, h, w):
        """在 token 形式和二维特征图形式之间切换以完成卷积。"""
        # 逐 token 扩展通道并进行非线性变换。
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        # [B,N,C] -> [B,C,H,W]，深度卷积后恢复为 [B,N,C] 并作残差相加。
        x = x + self.dwc(x.reshape(x.shape[0], h, w, x.shape[-1]).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # 再激活并投影到期望的输出通道数。
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvLayer(nn.Module):
    """按 ``Dropout -> Conv2d -> Norm -> Activation`` 组合的通用卷积层。"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()

        # 传入 None 可分别关闭 dropout、归一化或激活函数。
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """依次执行构造时启用的各个子层。"""
        if self.dropout is not None:
            x = self.dropout(x)

        # 卷积是该包装层中唯一始终执行的运算。
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class RoPE(torch.nn.Module):
    r"""二维旋转位置编码（Rotary Positional Embedding）。

    构造时为特征图每个 ``(h,w)`` 位置预计算一组正弦/余弦旋转量；
    前向时把相邻两维特征视为复数的实部和虚部，通过复数乘法注入位置。
    """

    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()

        # shape 通常为 [H,W,C]；前两维是空间轴，最后一维是特征维。
        channel_dims, feature_dim = shape[:-1], shape[-1]

        # 每个空间轴分配相同数量的旋转频率。
        k_max = feature_dim // (2 * len(channel_dims))

        # 确保特征维能完整拆成复数对及各空间轴的频率分组。
        assert feature_dim % k_max == 0

        # 生成从低频到高频的角速度，并与每个二维坐标相乘得到旋转角。
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        # 将 cos/sin 作为复数实部/虚部；形状最终可广播到输入特征。
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)

        # buffer 会随模型迁移设备、写入 state_dict，但不会作为可训练参数更新。
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        """将输入特征按二维位置旋转，并返回与输入相同的实数形状。"""
        # PyTorch 的复数视图和三角计算在 float32 下更稳定。
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        # 最后一维每两个数配成一个复数，然后乘以对应位置的单位复数。
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x

        # 拆回实部/虚部并展平，恢复原始特征维布局。
        return torch.view_as_real(pe_x).flatten(-2)


class TTTBlock(nn.Module):
    r"""分层 ViTTT 中的基本编码块。

    数据依次经过条件位置编码（CPE）、预归一化 TTT 和预归一化 MLP，
    三部分都使用残差连接；RoPE 额外传给 TTT 以提供显式二维位置信息。

    参数:
        dim: 当前 stage 的特征维度。
        input_resolution: 当前 stage 的固定空间尺寸 ``(H,W)``。
        num_heads: TTT 内部的头数。
        mlp_ratio: MLP 隐藏维度相对 ``dim`` 的倍数。
        qkv_bias: 是否启用 q/k/v 偏置。
        drop: dropout 概率。
        drop_path: 随机深度概率。
        act_layer: 激活函数类型。
        norm_layer: 归一化层类型。
    """

    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # 深度卷积生成条件位置编码，不改变通道数或空间分辨率。
        self.cpe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        # TTT 分支：LayerNorm 后同时提供序列、空间尺寸和旋转位置编码。
        self.norm1 = norm_layer(dim)
        self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim))
        self.attn = TTT(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # FFN 分支使用本文件定义的“线性层 + 深度卷积”Mlp。
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        """在固定 ``H×W`` 网格上处理形状为 ``[B,H*W,C]`` 的 token。"""
        H, W = self.input_resolution
        B, L, C = x.shape

        # 显式检查 token 数，避免错误 reshape 后悄悄打乱空间位置。
        assert L == H * W, "input feature has wrong size"

        # 条件位置编码：[B,L,C] -> [B,C,H,W] -> [B,L,C]。
        x = x + self.cpe(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # TTT/序列建模残差分支。
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, self.rope))

        # 带局部深度卷积的前馈网络残差分支。
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

    def extra_repr(self) -> str:
        """让 ``print(model)`` 显示该块最关键的结构参数。"""
        return f"dim={self.dim}, input_resolution={self.input_resolution}, " \
               f"mlp_ratio={self.mlp_ratio}"


class PatchMerging(nn.Module):
    r"""stage 之间的空间下采样与通道投影层。

    输入为 ``[B,H*W,dim]``，输出为
    ``[B,ceil(H/2)*ceil(W/2),dim_out]``。这里用倒残差风格的
    ``1×1 扩展 -> 3×3 深度卷积下采样 -> 1×1 投影`` 代替直接拼接 patch。

    参数:
        input_resolution: 输入特征的 ``(H,W)``。
        dim: 输入通道数。
        dim_out: 输出通道数。
        ratio: 深度卷积内部通道相对 ``dim_out`` 的扩展倍数。
    """

    def __init__(self, input_resolution, dim, dim_out, ratio=4.0):
        super().__init__()
        self.input_resolution = input_resolution
        in_channels = dim
        out_channels = dim_out

        # 第一个 1×1 卷积扩展通道；中间深度卷积以 stride=2 完成下采样；
        # 最后一个 1×1 卷积投影到下一 stage 的通道数且不附加激活。
        self.conv = nn.Sequential(
            ConvLayer(in_channels, int(out_channels * ratio), kernel_size=1, norm=None),
            ConvLayer(int(out_channels * ratio), int(out_channels * ratio), kernel_size=3, stride=2, padding=1, groups=int(out_channels * ratio), norm=None),
            ConvLayer(int(out_channels * ratio), out_channels, kernel_size=1, act_func=None)
        )

    def forward(self, x):
        """把 token 序列还原成特征图，下采样后再展平为 token。"""
        H, W = self.input_resolution
        B, L, C = x.shape

        # 保证输入序列长度与该 stage 声明的空间分辨率一致。
        assert L == H * W, "input feature has wrong size"

        # [B,H*W,C] -> [B,C,H,W] -> 卷积下采样 -> [B,H'*W',C_out]。
        x = self.conv(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        return x


class BasicLayer(nn.Module):
    """分层主干中的一个完整 stage。

    一个 stage 先串行执行 ``depth`` 个同分辨率 TTTBlock，随后按需下采样。

    参数:
        dim/dim_out: 当前/下一 stage 的通道数。
        input_resolution: 当前 stage 的空间分辨率。
        depth: 当前 stage 内 TTTBlock 的数量。
        num_heads: 当前 stage 的 TTT 头数。
        mlp_ratio: MLP 通道扩展倍数。
        qkv_bias: 是否启用 q/k/v 偏置。
        drop/drop_path: dropout 和随机深度概率。
        norm_layer: 归一化层类型。
        downsample: stage 末尾使用的下采样层；None 表示保持尺度。
        use_checkpoint: 是否用重计算换取更低的训练显存占用。
    """

    def __init__(self, dim, dim_out, input_resolution, depth, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # 当前 stage 内所有块使用相同的分辨率、通道数和头数；
        # drop_path 可传入列表，从而让每个块使用不同的随机深度概率。
        self.blocks = nn.ModuleList([
            TTTBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                     mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])

        # 除最后一个 stage 外，通常在末尾创建 PatchMerging 进入下一尺度。
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, dim_out=dim_out)
        else:
            self.downsample = None

    def forward(self, x):
        """执行 stage 内的所有块，并在需要时下采样。"""
        for blk in self.blocks:
            if self.use_checkpoint:
                # 不保存块内中间激活，反向传播时重新计算，以时间换显存。
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        # 下采样发生在当前 stage 所有 TTTBlock 之后。
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        """为模型结构打印补充当前 stage 的关键信息。"""
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class Stem(nn.Module):
    r"""卷积式图像嵌入 Stem。

    两次 stride=2 卷积总计把空间尺寸缩小 4 倍，再输出 ``embed_dim`` 通道。
    因而默认 224×224 图像会得到 56×56 个 token。

    参数:
        img_size: 输入图像尺寸。
        patch_size: 期望的 patch/token 尺寸，用于记录输出网格大小。
        in_chans: 输入图像通道数。
        embed_dim: 首个 stage 的 token 特征维度。
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        # 保存 token 网格尺寸，后续各 stage 据此推导自己的 H、W。
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.patches_resolution = patches_resolution

        # 前三层提取浅层局部特征；第四层完成第二次下采样并扩展通道；
        # 最后的 1×1 卷积压缩到 embed_dim，不使用末端激活。
        self.conv = nn.Sequential(
            ConvLayer(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None)
        )

    def forward(self, x):
        """把 ``[B,in_chans,H,W]`` 图像转换成 ``[B,N,embed_dim]`` token。"""
        x = self.conv(x)

        # 合并空间维，再把通道维移到最后以适配 TTTBlock。
        x = x.flatten(2).transpose(1, 2)
        return x


class ViTTT(nn.Module):
    r"""四阶段分层 ViTTT 分类网络。

    典型数据流（默认输入 224×224）为：
    ``图像 -> Stem: 56² tokens -> stage0: 28² -> stage1: 14²
    -> stage2: 7² -> stage3 -> 全局池化 -> 分类``。
    箭头后的尺度变化发生在前三个 stage 末尾的 PatchMerging。

    参数:
        img_size/patch_size/in_chans: 输入及初始 token 化配置。
        num_classes: 分类类别数。
        dim: 各 stage 的通道数列表。
        depths: 各 stage 的 TTTBlock 数量列表。
        num_heads: 各 stage 的 TTT 头数列表。
        mlp_ratio: MLP 隐藏通道扩展倍数。
        qkv_bias: 是否启用 q/k/v 偏置。
        drop_rate/drop_path_rate: dropout 和最大随机深度概率。
        norm_layer: TTTBlock 内使用的归一化层类型。
        use_checkpoint: 是否开启梯度检查点。
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 dim=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, use_checkpoint=False, **kwargs):
        super().__init__()

        # 保存全局结构信息；num_layers 通常为 4。
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = dim[0]
        self.mlp_ratio = mlp_ratio

        # 卷积 Stem 输出第一个 stage 所需的 token 序列。
        self.patch_embed = Stem(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=dim[0])
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # 对刚生成的所有 token 做统一 dropout。
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 为全网络所有块生成从 0 到 drop_path_rate 线性递增的随机深度概率。
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 逐 stage 构建分层主干。
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 第 i 个 stage 的分辨率相对 Stem 输出缩小 2**i 倍。
            # 前三个 stage 在末尾下采样，并将通道投影到 dim[i+1]。
            layer = BasicLayer(dim=dim[i_layer],
                               dim_out=dim[i_layer + 1] if i_layer < self.num_layers - 1 else None,
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # token 序列转为 [B,C,N] 后用 BatchNorm，随后沿 N 维全局平均池化。
        self.norm = nn.BatchNorm1d(dim[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # 类别数不为正时保留最终特征，便于将网络用作骨干。
        self.head = nn.Linear(dim[-1], num_classes) if num_classes > 0 else nn.Identity()

        # 递归初始化所有已创建的子模块。
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """初始化线性层和 LayerNorm；卷积/BatchNorm 沿用 PyTorch 默认策略。"""
        if isinstance(m, nn.Linear):
            # 线性层权重采用 ViT 常用的截断正态分布，偏置清零。
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # LayerNorm 初始为恒等仿射变换。
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        """声明不做权重衰减的传统绝对位置编码参数名，供训练脚本查询。"""
        return {'absolute_pos_embed'}

    def forward_features(self, x):
        """执行 Stem、四个 stage、归一化与全局池化，返回图像级特征。"""
        # 图像 [B,3,H,W] -> 初始 token [B,N0,C0]。
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        # 每个 stage 在内部建模；前三个 stage 结束时还会下采样。
        for layer in self.layers:
            x = layer(x)

        # BatchNorm1d 以通道为第二维，因此先从 [B,N,C] 转为 [B,C,N]。
        x = self.norm(x.transpose(1, 2))

        # 沿 token 维做全局平均池化：[B,C,N] -> [B,C,1]。
        x = self.avgpool(x)

        # 移除长度为 1 的空间维，得到每张图像一个 C 维向量。
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        """完整前向：提取图像级特征并送入分类头。"""
        x = self.forward_features(x)
        x = self.head(x)
        return x


def h_vittt_tiny(**kwargs):
    """构建 tiny 规格：各 stage 深度为 [1,3,9,4]。"""
    # 通道数和头数同步增长，保证各 stage 的单头维度合理。
    model = ViTTT(dim=[64, 128, 320, 512], depths=[1, 3, 9, 4], num_heads=[2, 4, 10, 16], **kwargs)
    return model


def h_vittt_small(**kwargs):
    """构建 small 规格：沿用 tiny 的通道配置，但增加每个 stage 的块数。"""
    model = ViTTT(dim=[64, 128, 320, 512], depths=[2, 6, 18, 8], num_heads=[2, 4, 10, 16], **kwargs)
    return model


def h_vittt_base(**kwargs):
    """构建 base 规格：在 small 深度基础上进一步提高各 stage 通道数。"""
    model = ViTTT(dim=[96, 192, 448, 640], depths=[2, 6, 18, 8], num_heads=[3, 6, 14, 20], **kwargs)
    return model
