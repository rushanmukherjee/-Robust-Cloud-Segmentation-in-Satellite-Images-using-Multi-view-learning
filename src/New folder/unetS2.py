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
class CloudSegmentation1(nn.Module):
    def __init__(
            self,  n_input_channels: int = 13,
        ) -> None:

        super(CloudSegmentation1, self).__init__()

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

        self.oplayer = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=1)


    ## Forward Function for Unet
    def forward(self, s2: torch.Tensor) -> Dict[str, torch.Tensor]:

        ##Encoder
        enc1 = self.enc1(s2)
        #print(f"double conv1 shape {enc1.shape}")
        encP1 = self.pool(enc1)
        #print(f"Pool 1 shape {encP1.shape}")

        enc2 = self.enc2(encP1)
        #print(f"double conv2 shape {enc2.shape}")
        encP2 = self.pool(enc2)
        #print(f"Pool 2 shape {encP2.shape}")

        enc3 = self.enc3(encP2)
        #print(f"double conv3 shape {enc3.shape}")
        encP3 = self.pool(enc3)
        #print(f"Pool 3 shape {encP3.shape}")

        enc4 = self.enc4(encP3)
        #print(f"double conv4 shape {enc4.shape}")
        encP4 = self.pool(enc4)
        #print(f"Pool 4 shape {encP4.shape}")

        bottleneck = self.bottleneck(encP4)
        #print(f"bottleneck shape is {bottleneck.shape}")

        ##Decoder
        dec_up4 = self.dec_upsample4(bottleneck, enc4)
        #print(f"Dec upsampling 4 shape {dec_up4.shape}")

        dec_up3 = self.dec_upsample3(dec_up4, enc3)
        #print(f"Dec upsampling 3 shape {dec_up3.shape}")

        dec_up2 = self.dec_upsample2(dec_up3, enc2)
        #print(f"Dec upsampling 2 shape {dec_up2.shape}")

        dec_up1 = self.dec_upsample1(dec_up2, enc1)
        #print(f"Dec upsampling 1 shape is {dec_up1.shape}")

        output = self.oplayer(dec_up1)
        #print(f"output tensor shape {output.shape}")


        return {'cloud_labels' : output}
        
def main():
    # Visualize architecture using torchview
    arch = CloudSegmentation1()
    input_size = (
        (10,13,512,512)
    )
    arch_graph = draw_graph(
        arch, 
        input_size=input_size,
        device='meta',
        graph_dir='LR',
        expand_nested=True,
        graph_name='Cloud Segmentation1',
        save_graph=True,
        directory='./models/architectures/arch_vis/'
    )
    arch_graph.visual_graph
    
if __name__ == '__main__':
    main()
