from typing import Union, Dict, List, Tuple
import torch
from torch import nn
from torchview import draw_graph
from torch.nn.functional import leaky_relu
from torch.nn.functional import batch_norm

## Define typical double convolution block
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out, dropout_rate = 0.1):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)
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
class UnetS1_adaptiveDropout(nn.Module):
    def __init__(
            self,  S1_input_channels: int = 3,
        ) -> None:

        super(UnetS1_adaptiveDropout, self).__init__()

        dropout_rates = {
                'encoder': [0.1, 0.1, 0.2, 0.2, 0.3],  # Increasing dropout in deeper layers
                'decoder': [0.3, 0.2, 0.2, 0.1, 0]        # Decreasing dropout towards output
            }

        ##Pooling Operations
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        ##Encoder Operations
        self.S1enc1 = conv_block(ch_in=S1_input_channels,ch_out=16, dropout_rate=dropout_rates['encoder'][0])
        self.S1enc2 = conv_block(ch_in=16,ch_out=32, dropout_rate=dropout_rates['encoder'][1])
        self.S1enc3 = conv_block(ch_in=32,ch_out=64, dropout_rate=dropout_rates['encoder'][2])
        self.S1enc4 = conv_block(ch_in=64,ch_out=128, dropout_rate=dropout_rates['encoder'][3])
        self.S1enc5 = conv_block(ch_in=128,ch_out=256, dropout_rate=dropout_rates['encoder'][4])

        ##Decoder operations
        self.dec_upsample5 = up_conv(ch_in=256,ch_out=256)
        self.dec_conv5 = conv_block(ch_in=512,ch_out=128, dropout_rate= dropout_rates['decoder'][0])
        self.dec_upsample4 = up_conv(ch_in=128,ch_out=128)
        self.dec_conv4 = conv_block(ch_in=256,ch_out=64, dropout_rate = dropout_rates['decoder'][1])
        self.dec_upsample3 = up_conv(ch_in=64,ch_out=64)
        self.dec_conv3 = conv_block(ch_in=128,ch_out=32, dropout_rate = dropout_rates['decoder'][2])
        self.dec_upsample2 = up_conv(ch_in=32,ch_out=32)
        self.dec_conv2 = conv_block(ch_in=64,ch_out=16, dropout_rate = dropout_rates['decoder'][3])
        self.dec_upsample1 = up_conv(ch_in=16,ch_out=16)
        self.dec_conv1 = conv_block(ch_in=32,ch_out=16, dropout_rate=dropout_rates['decoder'][4])

        self.oplayer = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1)


    ## Forward Function for Unet
    def forward(self, s1: torch.Tensor) -> Dict[str, torch.Tensor]:

        ##Encoder
        enc1 = self.S1enc1(s1)
        encP1 = self.pool(enc1)
        enc2 = self.S1enc2(encP1)
        encP2 = self.pool(enc2)
        enc3 = self.S1enc3(encP2)
        encP3 = self.pool(enc3)
        enc4 = self.S1enc4(encP3)
        encP4 = self.pool(enc4)
        enc5 = self.S1enc5(encP4)
        encP5 = self.pool(enc5)
        
        ##Decoder
        dec_up5 = self.dec_upsample5(encP5)
        dec_cat5 = torch.cat((dec_up5,enc5),dim=1)
        dec_conv5 = self.dec_conv5(dec_cat5)
        dec_up4 = self.dec_upsample4(dec_conv5)
        dec_cat4 = torch.cat((dec_up4,enc4),dim=1)
        dec_conv4 = self.dec_conv4(dec_cat4)
        dec_up3 = self.dec_upsample3(dec_conv4)
        dec_cat3 = torch.cat((dec_up3,enc3),dim=1)
        dec_conv3 = self.dec_conv3(dec_cat3)
        dec_up2 = self.dec_upsample2(dec_conv3)
        dec_cat2 = torch.cat((dec_up2,enc2),dim=1)
        dec_conv2 = self.dec_conv2(dec_cat2)
        dec_up1 = self.dec_upsample1(dec_conv2)
        dec_cat1 = torch.cat((dec_up1,enc1),dim=1)
        dec_conv1 = self.dec_conv1(dec_cat1)

        output = self.oplayer(dec_conv1)

        return {'cloud_labels' : output}
        
def main():
    # Visualize architecture using torchview
    arch = UnetS1_adaptiveDropout()
    input_size = (
        (10,3,512,512)
    )
    arch_graph = draw_graph(
        arch, 
        input_size=input_size,
        device='meta',
        graph_dir='LR',
        expand_nested=True,
        graph_name='Unet S1 adaptive Dropout',
        save_graph=True,
        directory='./models/architectures/arch_vis/'
    )
    arch_graph.visual_graph
    
if __name__ == '__main__':
    main()
