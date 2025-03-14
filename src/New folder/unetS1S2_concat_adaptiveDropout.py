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
class CloudSegmentationS1S2_concat_adaptiveDropout(nn.Module):
    def __init__(
            self,  S2_input_channels: int = 13, S1_input_channels=3,
        ) -> None:

        super(CloudSegmentationS1S2_concat_adaptiveDropout, self).__init__()

        dropout_rates = {
                'encoder': [0.1, 0.1, 0.2, 0.2, 0.3],  # Increasing dropout in deeper layers
                'decoder': [0.3, 0.2, 0.2, 0.1]        # Decreasing dropout towards output
            }
        
        ##Pooling Operations
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        ##S2 Encoder Operations
        self.S2enc1 = conv_block(ch_in=S2_input_channels,ch_out=16, dropout_rate=dropout_rates['encoder'][0])
        self.S2enc2 = conv_block(ch_in=16,ch_out=32, dropout_rate=dropout_rates['encoder'][1])
        self.S2enc3 = conv_block(ch_in=32,ch_out=64, dropout_rate=dropout_rates['encoder'][2])
        self.S2enc4 = conv_block(ch_in=64,ch_out=128, dropout_rate=dropout_rates['encoder'][3])
        self.S2enc5 = conv_block(ch_in=128,ch_out=256, dropout_rate=dropout_rates['encoder'][4])
        
        ##S1 Encoder Operations
        self.S1enc1 = conv_block(ch_in=S1_input_channels,ch_out=16, dropout_rate=dropout_rates['encoder'][0])
        self.S1enc2 = conv_block(ch_in=16,ch_out=32, dropout_rate=dropout_rates['encoder'][1])
        self.S1enc3 = conv_block(ch_in=32,ch_out=64, dropout_rate=dropout_rates['encoder'][2])
        self.S1enc4 = conv_block(ch_in=64,ch_out=128, dropout_rate=dropout_rates['encoder'][3])
        self.S1enc5 = conv_block(ch_in=128,ch_out=256, dropout_rate=dropout_rates['encoder'][4])

        ##Decoder operations
        self.dec_upsample5 = up_conv(ch_in=512,ch_out=256)
        self.dec_conv5 = conv_block(ch_in=768,ch_out=128, dropout_rate=dropout_rates['decoder'][0])

        self.dec_upsample4 = up_conv(ch_in=128,ch_out=128)
        self.dec_conv4 = conv_block(ch_in=384,ch_out=64, dropout_rate=dropout_rates['decoder'][1])

        self.dec_upsample3 = up_conv(ch_in=64,ch_out=64)
        self.dec_conv3 = conv_block(ch_in=192,ch_out=32, dropout_rate=dropout_rates['decoder'][2])

        self.dec_upsample2 = up_conv(ch_in=32,ch_out=32)
        self.dec_conv2 = conv_block(ch_in=96,ch_out=16, dropout_rate=dropout_rates['decoder'][3])

        self.dec_upsample1 = up_conv(ch_in=16,ch_out=16)
        self.dec_conv1 = conv_block(ch_in=48,ch_out=8)

        self.output = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)


    ## Forward Function for Unet
    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        ##Sentinel 2 Encoder
        S2enc1 = self.S2enc1(s2)
        #print(f"S2 level 1 shape is {S2enc1.shape}")
        S2encP1 = self.pool(S2enc1)
        S2enc2 = self.S2enc2(S2encP1)
        #print(f"S2 level 2 shape is {S2enc2.shape}")
        S2encP2 = self.pool(S2enc2)
        S2enc3 = self.S2enc3(S2encP2)
        #print(f"S2 level 3 shape is {S2enc3.shape}")
        S2encP3 = self.pool(S2enc3)
        S2enc4 = self.S2enc4(S2encP3)
        #print(f"S2 level 4 shape is {S2enc4.shape}")
        S2encP4 = self.pool(S2enc4)
        S2enc5 = self.S2enc5(S2encP4)
        #print(f"S2 level 5 shape is {S2enc5.shape}")
        S2encP5 = self.pool(S2enc5)
        
        ##Sentinel 1 Encoder
        S1enc1 = self.S1enc1(s1)
        #print(f"S1 level 1 shape is {S2enc1.shape}")
        S1encP1 = self.pool(S1enc1)
        S1enc2 = self.S1enc2(S1encP1)
        #print(f"S1 level 2 shape is {S2enc2.shape}")
        S1encP2 = self.pool(S1enc2)
        S1enc3 = self.S1enc3(S1encP2)
        #print(f"S1 level 3 shape is {S2enc3.shape}")
        S1encP3 = self.pool(S1enc3)
        S1enc4 = self.S1enc4(S1encP3)
        #print(f"S1 level 4 shape is {S2enc4.shape}")
        S1encP4 = self.pool(S1enc4)
        S1enc5 = self.S1enc5(S1encP4)
        #print(f"S1 level 5 shape is {S2enc5.shape}")
        S1encP5 = self.pool(S1enc5)

        ##Bottleneck concat
        bottleneck = torch.cat((S2encP5,S1encP5), dim=1)
        #print(f"bottleneck shape is {bottleneck.shape}")

        ##Decoder
        dec_up5 = self.dec_upsample5(bottleneck)
        #print(f"upsample 5 shape is {dec_up5.shape}")
        dec_cat5 = torch.cat((dec_up5,S1enc5,S2enc5), dim=1)
        dec_conv5 = self.dec_conv5(dec_cat5)
        #print(f"decoder conv shape is {dec_conv5.shape}")

        dec_up4 = self.dec_upsample4(dec_conv5)
        #print(f"upsample 4shape is {dec_up4.shape}")
        dec_cat4 = torch.cat((dec_up4,S1enc4,S2enc4), dim=1)
        dec_conv4 = self.dec_conv4(dec_cat4)
        #print(f"decoder conv shape is {dec_conv4.shape}")

        dec_up3 = self.dec_upsample3(dec_conv4)
        #print(f"upsample 3 shape is {dec_up3.shape}")
        dec_cat3 = torch.cat((dec_up3,S1enc3,S2enc3), dim=1)
        dec_conv3 = self.dec_conv3(dec_cat3)
        #print(f"decoder conv shape is {dec_conv3.shape}")

        dec_up2 = self.dec_upsample2(dec_conv3)
        #print(f"upsample 2 shape is {dec_up2.shape}")
        dec_cat2 = torch.cat((dec_up2,S1enc2,S2enc2), dim=1)
        dec_conv2 = self.dec_conv2(dec_cat2)
        #print(f"decoder conv shape is {dec_conv2.shape}")

        dec_up1 = self.dec_upsample1(dec_conv2)
        #print(f"upsample 1 shape is {dec_up1.shape}")
        dec_cat1 = torch.cat((dec_up1,S1enc1,S2enc1), dim=1)
        dec_conv1 = self.dec_conv1(dec_cat1)
        #print(f"decoder conv shape is {dec_conv1.shape}")

        output = self.output(dec_conv1)
        

        return {'cloud_labels' : output}
        
def main():
    # Visualize architecture using torchview
    arch = CloudSegmentationS1S2_concat_adaptiveDropout()
    input_size = (
        (13,1,512,512), (13,1,512,512)
    )
    arch_graph = draw_graph(
        arch, 
        input_size=input_size,
        device='meta',
        graph_dir='LR',
        expand_nested=True,
        graph_name='Cloud Segmentation S1S2 Concat',
        save_graph=True,
        directory='./models/architectures/arch_vis/'
    )
    arch_graph.visual_graph
    
if __name__ == '__main__':
    main()
