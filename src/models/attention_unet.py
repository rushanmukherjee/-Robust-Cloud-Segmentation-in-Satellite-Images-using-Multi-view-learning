from typing import Union, Dict, List, Tuple
import torch
from torch import nn
from torchview import draw_graph

## Refer to github link for full implementation of attention-unet
## https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py

## Define Attention Block
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
    
        return x*psi

## Define typical double convolution block
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

## Define upsampling and convolution combination block
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

# Architecture class defining the layers and forward method for the torch.nn.Module object
class Attention_Unet(nn.Module):
    """
    FlorestSegmentation2 class inherited from torch.nn.Module
    It can be summarized as:
        Inputs: DEM, time-series of rainfall
        Targets: binary segmentation of flooded/non-flooded pixels.
    This architecture contains: 
        two encoders for the two input variables
        one decoder for the flood segmenrtation.
    """
    def __init__(
            self,
            n_input_channels: int = 1 
        ) -> None:

        #number of filters increase by a factor of two while encoding and vice versa
        
        #num_filters = [64,128,256,512]

        """
        Constructor of the Unet class
        """
        super(Attention_Unet, self).__init__()
        
        ##Pooling operation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ## Encoder 1: DEM operations in linear format
        self.enc1 = conv_block(ch_in=n_input_channels,ch_out=8)
        self.enc2 = conv_block(ch_in=8,ch_out=16)
        self.enc3 = conv_block(ch_in=16,ch_out=32)
        self.enc4 = conv_block(ch_in=32,ch_out=64)
        self.enc5 = conv_block(ch_in=64,ch_out=128)

        # Encoder 2: Rain
        self.rain_encoder = nn.Sequential(
           nn.Linear(in_features=36, out_features=24),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=24, out_features=12),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=12, out_features=8),
            nn.ReLU(inplace=True),
        )

        #Decoder operations in linear format
        self.dec_upsample5 = up_conv(ch_in=128,ch_out=64)
        self.attn5 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.dec_conv5 = conv_block(ch_in=128,ch_out=64)

        self.dec_upsample4 = up_conv(ch_in=64,ch_out=32)
        self.attn4 = Attention_block(F_g=32,F_l=32,F_int=16)
        self.dec_conv4 = conv_block(ch_in=64,ch_out=32)

        self.dec_upsample3 = up_conv(ch_in=32,ch_out=16)
        self.attn3 = Attention_block(F_g=16,F_l=16,F_int=8)
        self.dec_conv3 = conv_block(ch_in=32,ch_out=16)

        self.dec_upsample2 = up_conv(ch_in=16,ch_out=8)
        self.attn2 = Attention_block(F_g=8,F_l=8,F_int=4)
        self.dec_conv2 = conv_block(ch_in=16,ch_out=8)


        self.oplayer = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
    
    def forward(self, dem: torch.Tensor, rain: torch.Tensor) -> Dict[str, torch.Tensor]:
                
        enc1 = self.enc1(dem)
        encP1 = self.pool(enc1)

        enc2 = self.enc2(encP1)
        encP2 = self.pool(enc2)

        enc3 = self.enc3(encP2)
        encP3 = self.pool(enc3)

        enc4 = self.enc4(encP3)
        encP4 = self.pool(enc4)

        enc5 = self.enc5(encP4)


        ##concatenation of rain data at bottleneck layer
        enc_rain = self.rain_encoder(rain)
        rain1=enc_rain.unsqueeze(2)
        rain2 = rain1.unsqueeze(2)
        half_size = enc5.shape[1]//2
        num_repeats = int(half_size/rain2.shape[1])
        rain_repeat = torch.repeat_interleave(rain2, num_repeats, dim=1)
        concat_1 = enc5[:,:half_size]*rain_repeat   ##multiplication rain fusion
        concat = torch.cat((concat_1, enc5[:,half_size:]),dim=1)

        #decoding operations and adding skip connections 
        dec_up5 = self.dec_upsample5(concat)
        dec_attn5 = self.attn5(g=dec_up5,x=enc4)
        dec_cat5 = torch.cat((dec_attn5,dec_up5), dim=1)
        dec_conv5 = self.dec_conv5(dec_cat5)

        dec_up4 = self.dec_upsample4(dec_conv5)
        dec_attn4 = self.attn4(g=dec_up4,x=enc3)
        dec_cat4 = torch.cat((dec_attn4,dec_up4), dim=1)
        dec_conv4 = self.dec_conv4(dec_cat4)

        dec_up3 = self.dec_upsample3(dec_conv4)
        dec_attn3 = self.attn3(g=dec_up3,x=enc2)
        dec_cat3 = torch.cat((dec_attn3,dec_up3), dim=1)
        dec_conv3 = self.dec_conv3(dec_cat3)
        
        dec_up2 = self.dec_upsample2(dec_conv3)
        dec_attn2 = self.attn2(g=dec_up2,x=enc1)
        dec_cat2 = torch.cat((dec_attn2,dec_up2), dim=1)
        dec_conv2 = self.dec_conv2(dec_cat2)

        output = self.oplayer(dec_conv2)

        return {
            'flood_labels': output, 
        }
        
def main():
    # Visualize architecture using torchview
    arch = Attention_Unet()
    input_size = (
        (5,1,2048,2048),
        (5,36)
    )
    arch_graph = draw_graph(
        arch, 
        input_size=input_size,
        device='meta',
        graph_dir='LR',
        expand_nested=True,
        graph_name='Florest_Segmentation_Attention_Unet',
        save_graph=True,
        directory='./models/architectures/arch_vis/'
    )
    arch_graph.visual_graph
    
if __name__ == '__main__':
    main()