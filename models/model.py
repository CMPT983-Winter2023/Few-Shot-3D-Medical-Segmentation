import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Tuple, Type, Union
from monai.networks.nets.swin_unetr import SwinTransformer, PatchMerging, PatchMergingV2
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

rearrange, _ = optional_import("einops", name="rearrange")
MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}

__all__ = [
    "SwinUNETR",
    "window_partition",
    "window_reverse",
    "WindowAttention",
    "SwinTransformerBlock",
    "PatchMerging",
    "PatchMergingV2",
    "MERGING_MODE",
    "BasicLayer",
    "SwinTransformer",
]


class CrossModalTransformer(nn.Module):
    def __init__(self, text_emb_dim, img_emb_dim, num_heads):
        super(CrossModalTransformer, self).__init__()
        self.text_emb_dim = text_emb_dim
        self.img_emb_dim = img_emb_dim
        self.num_heads = num_heads
        
        # Text embedding projection layer
        #self.text_projection = nn.Linear(text_emb_dim, img_emb_dim)
        # Image embedding projection layer
        #self.img_projection = nn.Linear(img_emb_dim, text_emb_dim)
        
        # Multi-head attention layers for text and image embeddings
        self.text_self_attention = nn.MultiheadAttention(text_emb_dim, num_heads, batch_first=True)
        self.img_self_attention = nn.MultiheadAttention(img_emb_dim, num_heads, batch_first=True)
        
        # Cross-modal attention layer
        self.cross_modal_attention_text = nn.MultiheadAttention(text_emb_dim, num_heads, batch_first=True)
        self.cross_modal_attention_image = nn.MultiheadAttention(img_emb_dim, num_heads, batch_first=True)
        
        # Layer normalization for all attention layers
        self.text_norm1 = nn.LayerNorm(img_emb_dim)
        self.img_norm1 = nn.LayerNorm(text_emb_dim)
        self.cross_modal_norm1_text = nn.LayerNorm(text_emb_dim)
        self.cross_modal_norm1_image = nn.LayerNorm(img_emb_dim)
        
        # Feedforward layers for all attention layers
        self.text_feedforward = nn.Sequential(
            nn.Linear(img_emb_dim, img_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(img_emb_dim * 4, img_emb_dim)
        )
        self.img_feedforward = nn.Sequential(
            nn.Linear(text_emb_dim, text_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(text_emb_dim * 4, text_emb_dim)
        )
        self.cross_modal_feedforward_text = nn.Sequential(
            nn.Linear(text_emb_dim, text_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(text_emb_dim * 4, text_emb_dim)
        )
        self.cross_modal_feedforward_image = nn.Sequential(
            nn.Linear(img_emb_dim, img_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(img_emb_dim * 4, img_emb_dim)
        )
        
        # Layer normalization for all feedforward layers
        self.text_norm2 = nn.LayerNorm(img_emb_dim)
        self.img_norm2 = nn.LayerNorm(text_emb_dim)
        self.cross_modal_norm2_text = nn.LayerNorm(text_emb_dim)
        self.cross_modal_norm2_image = nn.LayerNorm(img_emb_dim)
    
    def forward(self, text_emb, img_emb):
        # Project text embedding to image embedding space
        #text_emb_proj = self.text_projection(text_emb)
        
        # Self-attention on text and image embeddings
        #text_emb_self, _ = self.text_self_attention(text_emb_proj, text_emb_proj, text_emb_proj)
        #text_emb = self.text_norm1(text_emb_proj + text_emb_self)
        text_emb_self, _ = self.text_self_attention(text_emb, text_emb, text_emb)
        text_emb = self.text_norm1(text_emb + text_emb_self)
        text_emb = self.text_norm2(text_emb + self.text_feedforward(text_emb))
        
        img_emb_self, _ = self.img_self_attention(img_emb, img_emb, img_emb)
        img_emb = self.img_norm1(img_emb + img_emb_self)
        img_emb = self.img_norm2(img_emb + self.img_feedforward(img_emb))
        
        # Cross-modal attention between text and image embeddings
        cross_modal_emb, _ = self.cross_modal_attention_text(text_emb, img_emb, img_emb)
        text_emb = self.cross_modal_norm1_text(text_emb + cross_modal_emb)
        text_emb = self.cross_modal_norm2_text(text_emb + self.cross_modal_feedforward_text(text_emb))

        # Cross-modal attention between text and image embeddings
        cross_modal_emb, _ = self.cross_modal_attention_image(img_emb, text_emb, text_emb)
        img_emb = self.cross_modal_norm1_image(img_emb + cross_modal_emb)
        img_emb = self.cross_modal_norm2_image(img_emb + self.cross_modal_feedforward_image(img_emb))

        return text_emb, img_emb
        

class SwinUNETR(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        concat=False,
        contrast=False,
        attention=False
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).

        Examples::

            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)

            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))

            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)

        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize

        self.concat = concat
        self.contrast = contrast
        self.attention = attention

        if self.attention:
            self.cross_attention = CrossModalTransformer(
              text_emb_dim=27, 
              img_emb_dim=27, 
              num_heads=3)

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        if self.concat:
            self.decoder5 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=32 * feature_size,
                out_channels=8 * feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )
        else:
            self.decoder5 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=16 * feature_size,
                out_channels=8 * feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        if self.contrast:
            self.text_fc_contrast = nn.Linear(512, 16 * 27 * feature_size)
        if self.attention:
            self.text_fc_attention = nn.Linear(512, 16 * 27 * feature_size)
        if self.concat:
            self.text_fc_concat = nn.Linear(512, 16 * feature_size)

        self.lrelu = nn.LeakyReLU()

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

        self.cos_loss = nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')

    def load_from(self, weights):

        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(weights["state_dict"]["module.patch_embed.proj.weight"])
            self.swinViT.patch_embed.proj.bias.copy_(weights["state_dict"]["module.patch_embed.proj.bias"])
            for bname, block in self.swinViT.layers1[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")
            self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.reduction.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers2[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")
            self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.reduction.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers3[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers3")
            self.swinViT.layers3[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.reduction.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers4[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers4")
            self.swinViT.layers4[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.reduction.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.bias"]
            )

    def forward(self, x_in, text_in):
        #print(x_in.shape, text_in.shape)
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])

        dec4_shape = dec4.shape
        #print(dec4_shape)
        if self.contrast:
            flat = dec4.reshape(dec4_shape[0], -1)
            text_in_contrast = self.lrelu(self.text_fc_contrast(text_in))        
            L_contrast = self.cos_loss(text_in_contrast, flat, torch.ones((dec4_shape[0],), device=flat.device))
        else:
            L_contrast = None

        #print(flat.shape, text_in_contrast.shape)

        if self.attention and self.contrast:
            #print(text_in_contrast.shape, flat.shape)
            text_att, img_att = self.cross_attention(text_in_contrast.reshape(dec4_shape[0], dec4_shape[1], -1), flat.reshape(dec4_shape[0], dec4_shape[1], -1))
            dec4 = img_att.reshape(dec4_shape)
        elif self.attention:
            flat = dec4.reshape(dec4_shape[0], dec4_shape[1], -1)
            text_in_attention = self.lrelu(self.text_fc_attention(text_in))        
            text_att, img_att = self.cross_attention(text_in_attention.reshape(dec4_shape[0], dec4_shape[1], -1), flat)
            dec4 = img_att.reshape(dec4_shape)

        #print(text_att.shape, img_att.shape)

        if self.concat:
            if self.attention:
                text_in_concat = text_att.reshape(dec4_shape)
            else:
                text_in_concat = self.lrelu(self.text_fc_concat(text_in))      
                text_in_concat = text_in_concat.reshape(dec4_shape[0], dec4_shape[1], 1, 1, 1).repeat(1, 1, dec4_shape[2], dec4_shape[3], dec4_shape[4])
            concatenated = torch.cat((dec4, text_in_concat), dim=1)
            dec3 = self.decoder5(concatenated, hidden_states_out[3])
        else:    
            dec3 = self.decoder5(dec4, hidden_states_out[3])

        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        
        #print(logits.shape)

        return logits, L_contrast

class SSLHead(nn.Module):
    def __init__(self, args, upsample="vae", dim=768):
        super(SSLHead, self).__init__()
        patch_size = ensure_tuple_rep(2, args.spatial_dims)
        window_size = ensure_tuple_rep(7, args.spatial_dims)
        self.swinViT = SwinTransformer(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
        )
        self.rotation_pre = nn.Identity()
        self.rotation_head = nn.Linear(dim, 4)
        self.contrastive_pre = nn.Identity()
        self.contrastive_head = nn.Linear(dim, 512)
        if upsample == "large_kernel_deconv":
            self.conv = nn.ConvTranspose3d(dim, args.in_channels, kernel_size=(32, 32, 32), stride=(32, 32, 32))
        elif upsample == "deconv":
            self.conv = nn.Sequential(
                nn.ConvTranspose3d(dim, dim // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 4, dim // 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 8, dim // 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 16, args.in_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            )
        elif upsample == "vae":
            self.conv = nn.Sequential(
                nn.Conv3d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 2),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 4),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 8),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 8, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 16, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 16, args.in_channels, kernel_size=1, stride=1),
            )

    def forward(self, x):
        x_out = self.swinViT(x.contiguous())[4]
        #print(x_out.shape)
        _, c, h, w, d = x_out.shape
        x4_reshape = x_out.flatten(start_dim=2, end_dim=4)
        x4_reshape = x4_reshape.transpose(1, 2)
        x_rot = self.rotation_pre(x4_reshape[:, 0])
        x_rot = self.rotation_head(x_rot)
        x_contrastive = self.contrastive_pre(x4_reshape[:, 1])
        x_contrastive = self.contrastive_head(x_contrastive)
        x_rec = x_out.flatten(start_dim=2, end_dim=4)
        x_rec = x_rec.view(-1, c, h, w, d)
        #print(x_out.shape)
        x_rec = self.conv(x_rec)
        #print(x_out.shape)
        return x_rot, x_contrastive, x_rec