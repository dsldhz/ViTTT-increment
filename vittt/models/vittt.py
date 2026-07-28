# --------------------------------------------------------
# ViT^3: Unlocking Test-Time Training in Vision
# Modified from timm.models.vision_transformer
# Modified by Dongchen Han
# --------------------------------------------------------


""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

The official jax code is released and available at https://github.com/google-research/vision_transformer

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2021 Ross Wightman
"""

# 本文件实现“单尺度”ViTTT：
# 1. 先把输入图像切成固定大小的 patch，并映射为 token 序列；
# 2. 依次通过若干个带 TTT（Test-Time Training）模块的 Transformer Block；
# 3. 对所有空间 token 做平均池化，最后由分类头输出类别分数。
# 文件后半部分还保留了 timm/ViT 的权重初始化、预训练权重转换和模型注册逻辑。

# Python 标准库：数学运算、日志、函数参数固化、有序字典和对象复制。
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

# PyTorch 核心模块。
import torch
import torch.nn as nn
import torch.nn.functional as F

# timm 提供的数据预处理配置、模型构建工具和常用网络层。
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model

# ViTTT 的核心序列建模模块，接口与注意力层类似。
from .ttt_block import TTT

# 模块级日志器，主要用于记录位置编码缩放等预训练权重加载信息。
_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    """生成一份 timm 模型的默认配置。

    这里保存的不是网络结构，而是预训练权重地址、输入尺寸、归一化参数，
    以及 timm 在加载/替换分类器时需要知道的模块名称。
    """
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


# 不同 ViT/DeiT 预训练模型对应的元数据。
# ViTTT 的 tiny/small/base 工厂函数会复用相应 DeiT 配置，以便接入 timm 的构建流程。
default_cfgs = {
    # 标准 ViT：官方 Google JAX 实现导出的 ImageNet-1K 权重。
    'vit_tiny_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_tiny_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_base_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_base_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        ),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_large_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),

    # 标准 ViT：官方 Google JAX 实现导出的 ImageNet-21K 预训练权重。
    'vit_tiny_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_large_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        num_classes=21843),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz',
        num_classes=21843),
    'vit_huge_patch14_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz',
        hf_hub='timm/vit_huge_patch14_224_in21k',
        num_classes=21843),

    # DeiT：Facebook 发布的预训练权重。
    'deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0),
    'deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0,
        classifier=('head', 'head_dist')),

    # MILL 发布的 ImageNet-21K-P 预训练模型。
    'vit_base_patch16_224_miil_in21k': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/vit_base_patch16_224_in21k_miil.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear', num_classes=11221,
    ),
    'vit_base_patch16_224_miil': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm'
            '/vit_base_patch16_224_1k_miil_84_4.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear',
    ),
}


class Block(nn.Module):
    """单个 ViTTT 编码块。

    输入和输出形状均为 ``[B, N, C]``：
    ``B`` 是批大小，``N=H*W`` 是 patch 数量，``C`` 是特征维度。
    该模块由条件位置编码（CPE）、TTT 子层和 MLP 子层组成，
    三部分均通过残差连接保留原始信息。
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        # 深度卷积只在各自通道内聚合 3×3 邻域，用作随输入变化的条件位置编码。
        self.cpe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        # 第一个“预归一化 + TTT”分支。
        self.norm1 = norm_layer(dim)
        self.attn = TTT(dim, num_heads=num_heads, qkv_bias=qkv_bias)

        # DropPath 按样本随机丢弃整条残差分支；概率为 0 时不做任何变换。
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 第二个“预归一化 + MLP”分支。
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        """执行 CPE、TTT 和 MLP 三段残差计算。"""
        # 从 token 数 N 反推方形特征图边长；因此这里默认 N 是完全平方数。
        b, n, c = x.shape
        h = w = int(n ** 0.5)

        # [B,N,C] -> [B,C,H,W] 做深度卷积，再展平回 [B,N,C] 并残差相加。
        x = x + self.cpe(x.reshape(b, h, w, c).permute(0, 3, 1, 2)).flatten(2).transpose(1, 2)

        # TTT 需要显式接收二维网格尺寸，以便在序列与空间表示之间转换。
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))

        # 逐 token 的前馈网络进一步变换通道特征。
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """基于 TTT Block 的单尺度视觉 Transformer。

    整体数据流为：
    ``图像 [B,3,H,W] -> patch token [B,N,C] -> 多个 Block
    -> token 均值 [B,C] -> 分类 logits [B,num_classes]``。

    构造函数仍保留 timm 原始 ViT/DeiT 的部分兼容参数，例如
    ``distilled``、``representation_size`` 和不同初始化模式。
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """初始化网络结构。

        参数:
            img_size: 输入图像尺寸。
            patch_size: 每个 patch 的边长。
            in_chans: 输入图像通道数。
            num_classes: 分类类别数；小于等于 0 时分类头为恒等映射。
            embed_dim: 每个 token 的特征维度。
            depth: 编码块数量。
            num_heads: TTT 内部使用的头数。
            mlp_ratio: MLP 隐藏维度相对 ``embed_dim`` 的倍数。
            qkv_bias: 是否为 TTT 的 q/k/v 投影启用偏置。
            representation_size: 可选的分类前表示层维度。
            distilled: 是否创建 DeiT 蒸馏分类头（兼容参数）。
            drop_rate: MLP 等模块的 dropout 概率。
            attn_drop_rate: 注意力 dropout 概率（向下传给 Block）。
            drop_path_rate: 最深层的随机深度概率。
            embed_layer: 图像到 patch token 的嵌入层类型。
            norm_layer: 归一化层工厂。
            act_layer: 激活函数类型。
            weight_init: 权重初始化方案。
        """
        super().__init__()

        # 保存输出类别数和主干最终特征维度，供 timm 的统一接口读取。
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # 该字段沿用 DeiT 约定：普通模型 1 个特殊 token，蒸馏模型 2 个。
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # PatchEmbed 通常通过步长等于 patch_size 的卷积，把图像变为 [B,N,C]。
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # 随网络加深线性增大 DropPath 概率，浅层更稳定、深层正则更强。
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # 主干保持 token 数和通道数不变，是“单尺度”结构。
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

        # 所有编码块结束后再做一次归一化。
        self.norm = norm_layer(embed_dim)

        # 可选的分类前表示层；蒸馏模型按原始 DeiT 约定不使用该层。
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # 主分类头；当 num_classes <= 0 时可把模型当作特征提取器。
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # 可选的蒸馏头，用于兼容 DeiT 的双头结构。
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # 在所有子模块创建完成后统一初始化参数。
        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        """按 timm/ViT 约定初始化全模型参数。

        ``jax`` 表示尽量复现官方 JAX 初始化；``nlhb`` 会把分类头偏置
        初始化为 ``-log(num_classes)``，其余模式采用常见的截断正态分布。
        """
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        if mode.startswith('jax'):
            # JAX 兼容路径会根据完整模块名区分分类头、pre_logits 和普通层。
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            # 默认路径只依据模块类型初始化。
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        """兼容下游代码直接调用 ``model._init_weights`` 的包装方法。"""
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        """从 Google/Flax 风格的 ``.npz`` 文件加载预训练参数。"""
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        """返回传统 ViT 中通常不参与权重衰减的参数名。"""
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        """获取分类头；蒸馏模式下返回主头和蒸馏头。"""
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        """为迁移学习重新创建指定输出维度的分类头。"""
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        """提取整张图像的全局特征，不经过最终分类头。"""
        # [B,3,H,W] -> [B,N,C]。
        x = self.patch_embed(x)

        # 所有 Block 都保持 [B,N,C] 的形状不变。
        x = self.blocks(x)
        x = self.norm(x)

        # 不依赖 cls token，直接对 N 个空间 token 做全局平均池化。
        return x.mean(dim=1)

    def forward(self, x):
        """完整前向：先抽取全局特征，再输出分类 logits。"""
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """根据层类型和所选模式初始化 ViT 权重。

    不传 ``name/head_bias/jax_impl`` 时使用 timm 早期 ViT 的默认策略；
    传入模块名且 ``jax_impl=True`` 时，则尽量匹配官方 JAX 实现。
    """
    if isinstance(module, nn.Linear):
        # 分类头从零权重开始，偏置可使用负对数类别先验。
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)

        # 可选的表示层采用 LeCun 正态初始化。
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                # JAX 路径：线性权重使用 Xavier，MLP 偏置保留极小随机扰动。
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                # timm 默认路径：线性权重使用标准差 0.02 的截断正态分布。
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # 仅 JAX 兼容模式主动覆盖卷积层的 PyTorch 默认初始化。
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        # 归一化层初始为恒等变换：bias=0、scale=1。
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """加载 Google Brain Flax/JAX 实现导出的 ``.npz`` 权重。

    该函数是从 timm 的标准 ViT 兼容代码沿用而来，负责处理 NumPy 与
    PyTorch 的权重布局差异、混合 CNN 主干，以及位置编码尺寸变化。
    """
    import numpy as np

    def _n2p(w, t=True):
        """把 NumPy 权重转为 PyTorch 张量，并按层类型调整轴顺序。"""
        # 1×1×1 卷积形式的权重实际表示一维向量。
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()

        # Flax 卷积为 [H,W,I,O]、线性层为 [I,O]；PyTorch 顺序与其不同。
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    # 延迟导入并读取 npz，避免普通模型构建时强制依赖 NumPy 加载逻辑。
    w = np.load(checkpoint_path)

    # 某些优化器检查点把真实模型参数放在 opt/target/ 命名空间下。
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # 混合模型：先把 CNN 主干（stem 和各 stage）逐层拷贝进来。
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        # 纯 ViT：必要时把预训练输入卷积适配到当前输入通道数。
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))

    # 加载 patch 嵌入层和传统 ViT 的特殊 token。
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        # 输入分辨率变化时，插值空间位置编码以匹配新的 patch 网格。
        pos_embed_w = resize_pos_embed(
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        # 仅当当前模型和检查点都含表示层时才加载 pre_logits。
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        # 按编码块编号映射 LayerNorm、qkv、输出投影和两层 MLP 参数。
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    """把预训练位置编码的空间网格插值到当前模型尺寸。

    特殊 token 对应的位置向量保持不变，只对二维 patch 网格部分做双线性插值。
    逻辑改编自 Google Vision Transformer 的检查点加载代码。
    """
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]

    # 先将 cls/dist 等特殊 token 与规则的空间 token 分开。
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    # 旧位置编码默认来自方形网格；未显式给新尺寸时也按方形网格推断。
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)

    # [N,C] -> [1,C,H,W]，在空间维插值后再恢复成 token 序列。
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)

    # 把未插值的特殊 token 拼回序列开头。
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """把旧检查点中的权重布局转换为当前模型可加载的形式。

    主要兼容两类差异：旧式“手动 patchify + 线性层”的二维权重，
    以及预训练/当前输入尺寸不同时的位置编码。
    """
    out_dict = {}
    if 'model' in state_dict:
        # DeiT 检查点通常在顶层 ``model`` 字段中保存真正的 state_dict。
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # 把二维线性投影权重恢复为 PyTorch Conv2d 的 [O,I,H,W]。
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # 输入分辨率不同，需缩放位置编码的空间网格。
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)

        # 未命中特殊转换规则的参数原样保留。
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    """通过 timm 的统一构建器创建某个 ViTTT 规格。"""
    # 未由调用者覆盖时，根据规格名查找输入预处理和预训练权重配置。
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        # 本实现没有提供 timm 多层特征抽取器所需的 feature_info 接口。
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # 处理 ImageNet-21K 预训练模型可能带有 representation layer 的情况。
    default_num_classes = default_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # 微调类别数变化时默认移除原预训练表示层，避免其限制新的分类头。
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    # build_model_with_cfg 统一负责实例化、加载预训练权重和应用过滤函数。
    model = build_model_with_cfg(
        VisionTransformer, variant, pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in default_cfg['url'],
        **kwargs)
    return model


@register_model
def vittt_tiny(pretrained=False, **kwargs):
    """注册 tiny 规格：patch=16、通道=192、12 层、6 个头。"""
    # kwargs 可覆盖类别数、输入尺寸、dropout 等通用构造参数。
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('deit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vittt_small(pretrained=False, **kwargs):
    """注册 small 规格：patch=16、通道=384、12 层、6 个头。"""
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('deit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vittt_base(pretrained=False, **kwargs):
    """注册 base 规格：patch=16、通道=768、12 层、12 个头。"""
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('deit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model
