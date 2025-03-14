from typing import Union, Dict, List, Tuple
import torch
from torch import nn
from torchview import draw_graph
from torch.nn.functional import leaky_relu
from torch.nn.functional import batch_norm

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
class upSample(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(upSample,self).__init__()
        self.up = nn.ConvTranspose2d(ch_in, ch_in//2, kernel_size=2, stride=2)
        self.conv = conv_block(ch_in, ch_out)

    def forward(self,x1,x2):
        x1 = self.up(x1)
        x = torch.cat((x1,x2), dim=1)
        return self.conv(x)

# Architecture class defining the layers and forward method for the torch.nn.Module object
class S2_CloudSeg_allpretext(nn.Module):
    def __init__(
            self,  n_input_channels: int = 13,
        ) -> None:

        super(S2_CloudSeg_allpretext, self).__init__()

        ##Pooling Operations
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        ##Encoder Operations
        self.enc1 = conv_block(ch_in=n_input_channels,ch_out=16)
        self.enc2 = conv_block(ch_in=16,ch_out=32)
        self.enc3 = conv_block(ch_in=32,ch_out=64)
        self.enc4 = conv_block(ch_in=64,ch_out=128)

        ##Bottleneck
        self.bottleneck = conv_block(ch_in=128,ch_out=256)

        ##Decoder operations
        self.dec_upsample4 = upSample(ch_in=256,ch_out=128)
        self.dec_upsample3 = upSample(ch_in=128,ch_out=64)
        self.dec_upsample2 = upSample(ch_in=64,ch_out=32)
        self.dec_upsample1 = upSample(ch_in=32,ch_out=16)

        self.cloudseg_oplayer = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=1)

        self.landcover_oplayer = nn.Conv2d(in_channels=16, out_channels=11, kernel_size=1)

        self.elevation_oplayer = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

        self.S1_oplayer = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)


    ## Forward Function for Unet
    def forward(self, s2: torch.Tensor) -> Dict[str, torch.Tensor]:

        ##Encoder
        enc1 = self.enc1(s2)
        encP1 = self.pool(enc1)
        enc2 = self.enc2(encP1)
        encP2 = self.pool(enc2)
        enc3 = self.enc3(encP2)
        encP3 = self.pool(enc3)
        enc4 = self.enc4(encP3)
        encP4 = self.pool(enc4)

        bottleneck = self.bottleneck(encP4)

        ##Decoder Cloudseg
        decCS_up4 = self.dec_upsample4(bottleneck, enc4)
        decCS_up3 = self.dec_upsample3(decCS_up4, enc3)
        decCS_up2 = self.dec_upsample2(decCS_up3, enc2)
        decCS_up1 = self.dec_upsample1(decCS_up2, enc1)
        cloudseg_output = self.cloudseg_oplayer(decCS_up1)

        ##Decoder Landcover
        decLC_up4 = self.dec_upsample4(bottleneck, enc4)
        decLC_up3 = self.dec_upsample3(decLC_up4, enc3)
        decLC_up2 = self.dec_upsample2(decLC_up3, enc2)
        decLC_up1 = self.dec_upsample1(decLC_up2, enc1)
        landcover_output = self.landcover_oplayer(decLC_up1)

        ##Decoder Elevation
        decEL_up4 = self.dec_upsample4(bottleneck, enc4)
        decEL_up3 = self.dec_upsample3(decEL_up4, enc3)
        decEL_up2 = self.dec_upsample2(decEL_up3, enc2)
        decEL_up1 = self.dec_upsample1(decEL_up2, enc1)
        elevation_output = self.elevation_oplayer(decEL_up1)

        ##Decoder S1
        decS1_up4 = self.dec_upsample4(bottleneck, enc4)
        decS1_up3 = self.dec_upsample3(decS1_up4, enc3)
        decS1_up2 = self.dec_upsample2(decS1_up3, enc2)
        decS1_up1 = self.dec_upsample1(decS1_up2, enc1)
        S1_output = self.S1_oplayer(decS1_up1)

        return {'cloud_labels' : cloudseg_output,
                'landcover' : landcover_output,
                'elevation' : elevation_output,
                's1' : S1_output
        }
        
def main():
    # Visualize architecture using torchview
    arch = S2_CloudSeg_allpretext()
    input_size = (
        (10,13,512,512)
    )
    arch_graph = draw_graph(
        arch, 
        input_size=input_size,
        device='meta',
        graph_dir='LR',
        expand_nested=True,
        graph_name='S2 Cloud Segmentation all Pretext',
        save_graph=True,
        directory='./models/architectures/arch_vis/'
    )
    arch_graph.visual_graph
    
if __name__ == '__main__':
    main()
