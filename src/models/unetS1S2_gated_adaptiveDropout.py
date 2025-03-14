from typing import Union, Dict, List, Tuple
import torch
from torch import nn
import numpy as np
from torchview import draw_graph
from torch.nn.functional import leaky_relu
from torch.nn.functional import batch_norm

class GatedFusionModule(nn.Module):
    def __init__(
        self,
        enc_channel_list,
    ):
        super(GatedFusionModule, self).__init__()
        self.enc_channel_list = enc_channel_list

        self.num_features = len(enc_channel_list)
        
        num_ip_channels = np.sum(self.enc_channel_list)

        #self.h = 0
    
        self.gate_layer = nn.Sequential(
            nn.Linear(
                in_features=num_ip_channels,
                out_features=self.num_features, 
                bias=True
            ),
            nn.Softmax(dim=-1),
        )
        
    def forward(self, sensor_list: List[torch.Tensor]) -> torch.Tensor:
        
        # for sensor in sensor_list:
        #     sen_permute = torch.permute(sensor, [0,2,3,1])        
        #h_concat = torch.cat((h_s1, h_s2), dim=-1)
        # print(f"Dimensions of h_concat = {h_concat.shape}")
        
        # print(f"Dimensions of z = {z.shape}")
        
        concat_list = torch.cat(sensor_list, dim=1)
        #print(f"Dimensions of concat_list = {concat_list.shape}")
        list_permute = torch.permute(concat_list, [0,2,3,1])
        z = self.gate_layer(list_permute)

        z = torch.permute(z, [0,3,1,2])

        #h = torch.empty((concat_list.shape), device=concat_list.device) 
        # print(f"Dimensions of h = {h.shape}")

        h=0 
        #print(f"Dimensions of z = {z.shape}")
        #print(z[0,0,0,:])
        

        ##Print the dimensions of z and sensor_list 
        for idx,sensor in enumerate(sensor_list):
            #print(h)
            #print(f"Dimensions of idx = {idx}")
            #print(f"Dimensions of sensor = {sensor.shape}")

            weight = z[:,idx,:,] #16x16x16
            weight = torch.unsqueeze(weight, dim=1) #16x1x16x16
            #print(f"Dimensions of weight = {weight.shape}")

            h += weight*sensor
            # h = torch.cat((h, weight*sensor), dim=1)
            #print(f"Dimensions of h in loop = {h.shape}")

        #16x512x16x16
        #print(f"Dimensions of h = {self.h.shape}")
        
        return h



## Define typical double convolution block
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out, dropout_rate=0.2):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
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
class CloudSegmentationS1S2_gated_adaptiveDropout(nn.Module):
    def __init__(
            self,  S2_input_channels: int = 13, S1_input_channels=3,
        ) -> None:

        super(CloudSegmentationS1S2_gated_adaptiveDropout, self).__init__()
        
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

        ## Call the fusion module
        #self.bottleneck_list = [self.S1enc5, self.S2enc5]
        self.fusion_mod_bottleneck = GatedFusionModule([256,256])

        ##Decoder operations
        self.dec_upsample5 = up_conv(ch_in=256,ch_out=256)
        self.fusion_mod5 = GatedFusionModule([256,256,256])
        self.dec_conv5 = conv_block(ch_in=256,ch_out=128, dropout_rate=dropout_rates['decoder'][0])

        self.dec_upsample4 = up_conv(ch_in=128,ch_out=128)
        self.fusion_mod4 = GatedFusionModule([128,128,128])
        self.dec_conv4 = conv_block(ch_in=128,ch_out=64, dropout_rate=dropout_rates['decoder'][1])

        self.dec_upsample3 = up_conv(ch_in=64,ch_out=64)
        self.fusion_mod3 = GatedFusionModule([64,64,64])
        self.dec_conv3 = conv_block(ch_in=64,ch_out=32, dropout_rate=dropout_rates['decoder'][2])

        self.dec_upsample2 = up_conv(ch_in=32,ch_out=32)
        self.fusion_mod2 = GatedFusionModule([32,32,32])
        self.dec_conv2 = conv_block(ch_in=32,ch_out=16, dropout_rate=dropout_rates['decoder'][3])

        self.dec_upsample1 = up_conv(ch_in=16,ch_out=16)
        self.fusion_mod1 = GatedFusionModule([16,16,16])
        self.dec_conv1 = conv_block(ch_in=16,ch_out=8)

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

        ##Bottleneck Fusion
        bottleneck_list = [S1encP5, S2encP5]
        
        # print(f"S1encP5 shape is {S1encP5.shape}")
        # print(f"S2encP5 shape is {S2encP5.shape}")
        #print(f"bottleneck list shape is {bottleneck_list[0].shape}")

        fused_bottleneck = self.fusion_mod_bottleneck(bottleneck_list)
        #print(f"bottleneck shape is {fused_bottleneck.shape}")

        ##Decoder
        dec_up5 = self.dec_upsample5(fused_bottleneck)
        #print(f"upsample 5 shape is {dec_up5.shape}")
        dec_fusion5 = [dec_up5,S1enc5,S2enc5]
        dec_fusion5 = self.fusion_mod5(dec_fusion5)
        dec_conv5 = self.dec_conv5(dec_fusion5)
        #print(f"decoder conv shape is {dec_conv5.shape}")

        dec_up4 = self.dec_upsample4(dec_conv5)
        #print(f"upsample 4shape is {dec_up4.shape}")
        dec_fusion4 = [dec_up4,S1enc4,S2enc4]
        dec_fusion4 = self.fusion_mod4(dec_fusion4)
        dec_conv4 = self.dec_conv4(dec_fusion4)
        #print(f"decoder conv shape is {dec_conv4.shape}")

        dec_up3 = self.dec_upsample3(dec_conv4)
        #print(f"upsample 3 shape is {dec_up3.shape}")
        dec_fusion3 = [dec_up3,S1enc3,S2enc3]
        dec_fusion3 = self.fusion_mod3(dec_fusion3)
        dec_conv3 = self.dec_conv3(dec_fusion3)
        #print(f"decoder conv shape is {dec_conv3.shape}")

        dec_up2 = self.dec_upsample2(dec_conv3)
        #print(f"upsample 2 shape is {dec_up2.shape}")
        dec_fusion2 = [dec_up2,S1enc2,S2enc2]
        dec_fusion2 = self.fusion_mod2(dec_fusion2)
        dec_conv2 = self.dec_conv2(dec_fusion2)
        #print(f"decoder conv shape is {dec_conv2.shape}")

        dec_up1 = self.dec_upsample1(dec_conv2)
        #print(f"upsample 1 shape is {dec_up1.shape}")
        dec_fusion1 = [dec_up1,S1enc1,S2enc1]
        dec_fusion1 = self.fusion_mod1(dec_fusion1)
        dec_conv1 = self.dec_conv1(dec_fusion1)
        #print(f"decoder conv shape is {dec_conv1.shape}")

        output = self.output(dec_conv1)
        #print(f"output shape is {output.shape}")
        

        return {'cloud_labels' : output}
        
def main():
    # Visualize architecture using torchview
    arch = CloudSegmentationS1S2_gated_adaptiveDropout()
    input_size = (
        (13,1,512,512), (13,1,512,512)
    )
    arch_graph = draw_graph(
        arch, 
        input_size=input_size,
        device='meta',
        graph_dir='LR',
        expand_nested=True,
        graph_name='Cloud Segmentation Gated',
        save_graph=True,
        directory='./models/architectures/arch_vis/'
    )
    arch_graph.visual_graph
    
if __name__ == '__main__':
    main()
