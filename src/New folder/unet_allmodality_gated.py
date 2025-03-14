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
        #print(sensor_list[0].shape)
        concat_list = torch.cat(sensor_list, dim=1)
        #print(f"Dimensions of concat_list = {concat_list.shape}")
        list_permute = torch.permute(concat_list, [0,2,3,1])
        z = self.gate_layer(list_permute)

        z = torch.permute(z, [0,3,1,2])

        #h = torch.empty((concat_list.shape), device=concat_list.device) 
        # print(f"Dimensions of h = {h.shape}")

        h=torch.tensor([], device=torch.device('cuda')) 
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
            #print(f"Dimensions of h before summation = {h.shape}")
            #h += weight*sensor
            h = torch.cat((h, weight*sensor), dim=1)
            #print(f"Dimensions of h in loop = {h.shape}")

        #16x768x16x16
        
        return h


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
class CloudSeg_AllModality_Gated(nn.Module):
    def __init__(
            self,  S2_input_channels: int = 13, S1_input_channels=3, LC_input_channels=1, 
            azimuth_input_channels=1, wateroccurence_input_channels=1, DEM_input_channels=1,
        ) -> None:

        super(CloudSeg_AllModality_Gated, self).__init__()
        
        ##Pooling Operations
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        ##S2 Encoder Operations
        self.S2enc1 = conv_block(ch_in=S2_input_channels,ch_out=16)
        self.S2enc2 = conv_block(ch_in=16,ch_out=32)
        self.S2enc3 = conv_block(ch_in=32,ch_out=64)
        self.S2enc4 = conv_block(ch_in=64,ch_out=128)
        self.S2enc5 = conv_block(ch_in=128,ch_out=256)
        
        ##S1 Encoder Operations
        self.S1enc1 = conv_block(ch_in=S1_input_channels,ch_out=16)
        self.S1enc2 = conv_block(ch_in=16,ch_out=32)
        self.S1enc3 = conv_block(ch_in=32,ch_out=64)
        self.S1enc4 = conv_block(ch_in=64,ch_out=128)
        self.S1enc5 = conv_block(ch_in=128,ch_out=256)

        ##Landcover Encoder Operations
        self.LCenc1 = conv_block(ch_in=LC_input_channels,ch_out=4)
        self.LCenc2 = conv_block(ch_in=4,ch_out=8)
        self.LCenc3 = conv_block(ch_in=8,ch_out=16)
        self.LCenc4 = conv_block(ch_in=16,ch_out=32)
        self.LCenc5 = conv_block(ch_in=32,ch_out=64)

        ##DEM Encoder Operations
        self.DEMenc1 = conv_block(ch_in=DEM_input_channels,ch_out=4)
        self.DEMenc2 = conv_block(ch_in=4,ch_out=8)
        self.DEMenc3 = conv_block(ch_in=8,ch_out=16)
        self.DEMenc4 = conv_block(ch_in=16,ch_out=32)
        self.DEMenc5 = conv_block(ch_in=32,ch_out=64)

        ##Azimuth Encoder Operations
        self.Azienc1 = conv_block(ch_in=azimuth_input_channels,ch_out=4)
        self.Azienc2 = conv_block(ch_in=4,ch_out=8)
        self.Azienc3 = conv_block(ch_in=8,ch_out=16)
        self.Azienc4 = conv_block(ch_in=16,ch_out=32)
        self.Azienc5 = conv_block(ch_in=32,ch_out=64)

        ##Water Occurence Encoder Operations
        self.WOenc1 = conv_block(ch_in=wateroccurence_input_channels,ch_out=4)
        self.WOenc2 = conv_block(ch_in=4,ch_out=8)
        self.WOenc3 = conv_block(ch_in=8,ch_out=16)
        self.WOenc4 = conv_block(ch_in=16,ch_out=32)
        self.WOenc5 = conv_block(ch_in=32,ch_out=64)


        ## Call the fusion module
        #self.bottleneck_list = [self.S1enc5, self.S2enc5]
        self.fusion_mod_bottleneck = GatedFusionModule([256,256,64,64,64,64])

        ##Decoder operations
        self.dec_upsample5 = up_conv(ch_in=768,ch_out=256)
        self.fusion_mod5 = GatedFusionModule([256,256,256,64,64,64,64])
        self.dec_conv5 = conv_block(ch_in=1024,ch_out=128)

        self.dec_upsample4 = up_conv(ch_in=128,ch_out=128)
        self.fusion_mod4 = GatedFusionModule([128,128,128,32,32,32,32])
        self.dec_conv4 = conv_block(ch_in=512,ch_out=64)

        self.dec_upsample3 = up_conv(ch_in=64,ch_out=64)
        self.fusion_mod3 = GatedFusionModule([64,64,64,16,16,16,16])
        self.dec_conv3 = conv_block(ch_in=256,ch_out=32)

        self.dec_upsample2 = up_conv(ch_in=32,ch_out=32)
        self.fusion_mod2 = GatedFusionModule([32,32,32,8,8,8,8])
        self.dec_conv2 = conv_block(ch_in=128,ch_out=16)

        self.dec_upsample1 = up_conv(ch_in=16,ch_out=16)
        self.fusion_mod1 = GatedFusionModule([16,16,16,4,4,4,4])
        self.dec_conv1 = conv_block(ch_in=64,ch_out=8)

        self.cloud_output = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)


    ## Forward Function for Unet
    def forward(self, s1: torch.Tensor, s2: torch.Tensor, LC: torch.Tensor, 
                azimuth: torch.Tensor, WO: torch.Tensor, DEM: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        
        LC = LC.unsqueeze(1)
        DEM = DEM.unsqueeze(1)
        azimuth = azimuth.unsqueeze(1)
        WO = WO.unsqueeze(1)

        ## Encoder operations 
        S1enc1 = self.S1enc1(s1)
        S2enc1 = self.S2enc1(s2)
        LCenc1 = self.LCenc1(LC.float())
        DEMenc1 = self.DEMenc1(DEM)
        Azienc1 = self.Azienc1(azimuth)
        WOenc1 = self.WOenc1(WO)
        
        S1encP1 = self.pool(S1enc1)
        S1enc2 = self.S1enc2(S1encP1)
        S2encP1 = self.pool(S2enc1)
        S2enc2 = self.S2enc2(S2encP1)
        LCencP1 = self.pool(LCenc1)
        LCenc2 = self.LCenc2(LCencP1)
        DEMencP1 = self.pool(DEMenc1)
        DEMenc2 = self.DEMenc2(DEMencP1)
        AziencP1 = self.pool(Azienc1)
        Azienc2 = self.Azienc2(AziencP1)
        WOencP1 = self.pool(WOenc1)
        WOenc2 = self.WOenc2(WOencP1)

        S1encP2 = self.pool(S1enc2)
        S1enc3 = self.S1enc3(S1encP2)
        S2encP2 = self.pool(S2enc2)
        S2enc3 = self.S2enc3(S2encP2)
        LCencP2 = self.pool(LCenc2)
        LCenc3 = self.LCenc3(LCencP2)
        DEMencP2 = self.pool(DEMenc2)
        DEMenc3 = self.DEMenc3(DEMencP2)
        AziencP2 = self.pool(Azienc2)
        Azienc3 = self.Azienc3(AziencP2)
        WOencP2 = self.pool(WOenc2)
        WOenc3 = self.WOenc3(WOencP2)
                             

        S1encP3 = self.pool(S1enc3)
        S1enc4 = self.S1enc4(S1encP3)
        S2encP3 = self.pool(S2enc3)
        S2enc4 = self.S2enc4(S2encP3)
        LCencP3 = self.pool(LCenc3)
        LCenc4 = self.LCenc4(LCencP3)
        DEMencP3 = self.pool(DEMenc3)
        DEMenc4 = self.DEMenc4(DEMencP3)
        AziencP3 = self.pool(Azienc3)
        Azienc4 = self.Azienc4(AziencP3)
        WOencP3 = self.pool(WOenc3)
        WOenc4 = self.WOenc4(WOencP3)

        S1encP4 = self.pool(S1enc4)
        S1enc5 = self.S1enc5(S1encP4)
        S2encP4 = self.pool(S2enc4)
        S2enc5 = self.S2enc5(S2encP4)
        LCencP4 = self.pool(LCenc4)
        LCenc5 = self.LCenc5(LCencP4)
        DEMencP4 = self.pool(DEMenc4)
        DEMenc5 = self.DEMenc5(DEMencP4)
        AziencP4 = self.pool(Azienc4)
        Azienc5 = self.Azienc5(AziencP4)
        WOencP4 = self.pool(WOenc4)
        WOenc5 = self.WOenc5(WOencP4)

        S1encP5 = self.pool(S1enc5)
        S2encP5 = self.pool(S2enc5)
        LCencP5 = self.pool(LCenc5)
        DEMencP5 = self.pool(DEMenc5)
        AziencP5 = self.pool(Azienc5)
        WOencP5 = self.pool(WOenc5)
        
        ##Bottleneck Fusion
        bottleneck_list = [S1encP5, S2encP5, LCencP5, DEMencP5, AziencP5, WOencP5]

        fused_bottleneck = self.fusion_mod_bottleneck(bottleneck_list)
        #print(f"bottleneck shape is {fused_bottleneck.shape}")

        ##Decoder
        dec_up5 = self.dec_upsample5(fused_bottleneck)

        dec_fusion5_list = [dec_up5,S1enc5,S2enc5,LCenc5, DEMenc5, Azienc5, WOenc5]
        dec_fusion5 = self.fusion_mod5(dec_fusion5_list)
        dec_conv5 = self.dec_conv5(dec_fusion5)
        #print(f"decoder conv shape is {dec_conv5.shape}")

        dec_up4 = self.dec_upsample4(dec_conv5)
        #print(f"upsample 4shape is {dec_up4.shape}")
        dec_fusion4_list = [dec_up4,S1enc4,S2enc4,LCenc4, DEMenc4, Azienc4, WOenc4]
        dec_fusion4 = self.fusion_mod4(dec_fusion4_list)
        dec_conv4 = self.dec_conv4(dec_fusion4)
        #print(f"decoder conv shape is {dec_conv4.shape}")

        dec_up3 = self.dec_upsample3(dec_conv4)
        #print(f"upsample 3 shape is {dec_up3.shape}")
        dec_fusion3_list = [dec_up3,S1enc3,S2enc3,LCenc3, DEMenc3, Azienc3, WOenc3]
        dec_fusion3 = self.fusion_mod3(dec_fusion3_list)
        dec_conv3 = self.dec_conv3(dec_fusion3)
        #print(f"decoder conv shape is {dec_conv3.shape}")

        dec_up2 = self.dec_upsample2(dec_conv3)
        #print(f"upsample 2 shape is {dec_up2.shape}")
        dec_fusion2_list = [dec_up2,S1enc2,S2enc2,LCenc2, DEMenc2, Azienc2, WOenc2]
        dec_fusion2 = self.fusion_mod2(dec_fusion2_list)
        dec_conv2 = self.dec_conv2(dec_fusion2)
        #print(f"decoder conv shape is {dec_conv2.shape}")

        dec_up1 = self.dec_upsample1(dec_conv2)
        #print(f"upsample 1 shape is {dec_up1.shape}")
        dec_fusion1_list = [dec_up1,S1enc1,S2enc1,LCenc1, DEMenc1, Azienc1, WOenc1]
        dec_fusion1 = self.fusion_mod1(dec_fusion1_list)
        dec_conv1 = self.dec_conv1(dec_fusion1)
        #print(f"decoder conv shape is {dec_conv1.shape}")

        cloud_output = self.cloud_output(dec_conv1)
        #print(f"output shape is {output.shape}")
        

        return {'cloud_labels' : cloud_output}
        
def main():
    # Visualize architecture using torchview
    arch = CloudSeg_AllModality_Gated()
    input_size = (
        (13,1,512,512), (13,1,512,512), (13,1,512,512)
    )
    arch_graph = draw_graph(
        arch, 
        input_size=input_size,
        device='meta',
        graph_dir='LR',
        expand_nested=True,
        graph_name='Cloud Seg All Modality Gated',
        save_graph=True,
        directory='./models/architectures/arch_vis/'
    )
    arch_graph.visual_graph
    
if __name__ == '__main__':
    main()
