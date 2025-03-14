from typing import Union, Dict, List, Tuple 
import torch 
from torch import nn
import math
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from functools import partial
import torch.nn.functional as F
from collections import OrderedDict

# from do_conv_pytorch import DOConv2d
#

##Fusion Module
class FusionModule(nn.Module):
    def __init__(
            self,
            conv_channels,
            transformer_channels
    ):
        super(FusionModule, self).__init__()
        self.conv_channels = conv_channels
        self.transformer_channels = transformer_channels

##Decoder Module
class Decoder_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(Decoder_block,self).__init__()
        


## Define typical double convolution block
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.GELU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.GELU()
        )

    def forward(self,x):
        x = self.conv(x)
        return x

##MCANet class declaration 
class MCANet(nn.Module):

    def __init__(self,
                 n_input_channels: int=1, img_size=512, embed_dim=[96,240,384,384], num_heads=[1,2,4,8],
                  mlp_ratio=4., qkv_bias=True, qk_scale=None, depth=[1,2,3,2], sr_ratios=[4,2,2,1],
                 ) -> None:
        
        super(MCANet,self).__init__()
        
        ##Pooling operation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ##Encoder Convolution Layers
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels=n_input_channels, out_channels=64, kernel_size=7, stride=2),
                                    nn.BatchNorm2d(64), nn.GELU())
        self.enc2 = nn.Sequential(conv_block(ch_in=64,ch_out=64), conv_block(ch_in=64,ch_out=64), conv_block(ch_in=64,ch_out=64))

        self.enc3 = nn.Sequential(conv_block(ch_in=128,ch_out=128), conv_block(ch_in=128,ch_out=128), 
                                    conv_block(ch_in=128,ch_out=128), conv_block(ch_in=128,ch_out=128))
        
        self.enc4 = nn.Sequential(conv_block(ch_in=256,ch_out=256), conv_block(ch_in=256,ch_out=256), conv_block(ch_in=256,ch_out=256),
                                    conv_block(ch_in=256,ch_out=256), conv_block(ch_in=256,ch_out=256), conv_block(ch_in=256,ch_out=256))

        self.enc5 = nn.Sequential(conv_block(ch_in=512,ch_out=512), conv_block(ch_in=512,ch_out=512), conv_block(ch_in=512,ch_out=512))

        ##Guidance module for transformer
        self.guidance = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=48, kernel_size=1, stride=1),
                                 nn.GELU())
        
        ##Transformer layers
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed1 = PatchEmbed(
            img_size=img_size // 4, patch_size=4, in_chans=n_input_channels, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 16, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.patch_embed4 = PatchEmbed(
            img_size=img_size // 32, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3])


        self.blocks1 = nn.ModuleList([
            LGLBlock(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for i in range(depth[0])])
        
        self.blocks2 = nn.ModuleList([
            LGLBlock(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for i in range(depth[1])])
        
        self.blocks3 = nn.ModuleList([
            LGLBlock(
                dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for i in range(depth[2])])
        
        self.blocks4 = nn.ModuleList([
            LGLBlock(
                dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, sr_ratio=sr_ratios[3])
            for i in range(depth[3])])
        
        self.norm = nn.BatchNorm2d(embed_dim[-1])


        ##Decoder Layers

        #self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=1)
        
         

    def forward(self, sen1: torch.tensor, sen2: torch.tensor) -> torch.tensor:
        
        inp = torch.stack((sen1,sen2), dim=1)

        ##encoder operations 
        enc1 = self.enc1(inp)

        conv2 = self.enc2(enc1)
        print(f"shape of conv5 is {conv2.shape}")
        guidance = self.guidance(enc1)
        trans2 = self.blocks1(guidance)
        print(f"shape of trans5 is {trans2.shape}")
        pool2 = self.pool(conv2)

        conv3 = self.enc3(pool2)
        print(f"shape of conv5 is {conv3.shape}")
        trans3 = self.blocks2(trans2)
        print(f"shape of trans5 is {trans3.shape}")
        pool3 = self.pool(conv3)

        conv4 = self.enc4(pool3)
        print(f"shape of conv5 is {conv4.shape}")
        trans4 = self.blocks3(trans3)
        print(f"shape of trans5 is {trans4.shape}")
        pool4 = self.pool(conv4)

        conv5 = self.enc5(pool4)
        print(f"shape of conv5 is {conv5.shape}")
        trans5 = self.blocks4(trans4)
        print(f"shape of trans5 is {trans5.shape}")

        assert(False)

        return output




        