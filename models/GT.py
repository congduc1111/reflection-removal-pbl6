import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialChannelAttentionSub(nn.Module):
    """Cải tiến 1: Trừ đặc trưng thích nghi Adaptive Attention Subtraction."""
    def __init__(self, fea_ch):
        super(SpatialChannelAttentionSub, self).__init__()
        mid_ch = max(fea_ch // 4, 16)
        self.attn = nn.Sequential(
            nn.Conv2d(fea_ch * 2, mid_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, groups=mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, fea_ch, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x2_I, x3_R):
        alpha = self.attn(torch.cat([x2_I, x3_R], dim=1))
        return x2_I - alpha * x3_R

class MASK_GENE(nn.Module):
    """Cải tiến 2: Sinh Mask đa tỷ lệ với Depthwise Dilated Convolution."""
    def __init__(self, io_ch):
        super(MASK_GENE, self).__init__()
        self.io_ch = io_ch
        self.mask_conv0 = nn.Conv2d(io_ch * 3, io_ch * 2, kernel_size=1)
        self.mask_ac = nn.ReLU(inplace=True)
        self.spatial_ctx = nn.Sequential(
            nn.Conv2d(io_ch * 2, io_ch * 2, kernel_size=3, padding=2, dilation=2, groups=io_ch * 2),
            nn.BatchNorm2d(io_ch * 2, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.mask_conv1 = nn.Conv2d(io_ch * 2, io_ch * 2, kernel_size=1)
        
    def forward(self, x3_R, x2_I, x1_dec):
        x = torch.cat([x3_R, x2_I, x1_dec], dim=1)
        feat = self.mask_ac(self.mask_conv0(x))
        feat = feat + self.spatial_ctx(feat)
        mask = torch.sigmoid(self.mask_conv1(feat))
        mask_e, mask_d = torch.split(mask, self.io_ch, dim=1)
        return mask, mask_e, mask_d

class up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_num, fea_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.adaptive_sub = SpatialChannelAttentionSub(fea_ch)
        self.cv = nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(in_ch, track_running_stats=False)
        self.ac = nn.ReLU(inplace=True)
        self.pooling = torch.nn.AvgPool2d(3, stride=1, padding=1)
        self.mask_gene = MASK_GENE(fea_ch)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.BatchNorm2d(in_ch, track_running_stats=False), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.BatchNorm2d(in_ch, track_running_stats=False), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.BatchNorm2d(in_ch, track_running_stats=False), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch, track_running_stats=False), nn.ReLU(inplace=True),
        ) if conv_num == 4 else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch, track_running_stats=False), nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        f_diff = self.adaptive_sub(x2, x3)
        x = torch.cat([f_diff, x1], dim=1)
        mask, mask_e, mask_d = self.mask_gene(x3, x2, x1)     
        
        x = self.cv(x * mask)
        mask_avg = self.pooling(mask)
        x = x / torch.clamp(mask_avg, min=1e-6)
        x = self.bn(x)
        x = self.ac(x)
        return self.conv(x), mask, mask_e, mask_d

class Encoder(nn.Module):
    def __init__(self, encoder):
        super(Encoder, self).__init__()
        self.encoder = encoder
    def forward(self, x):
        return (self.encoder.features[0:4](x), self.encoder.features[4:9](x),
                self.encoder.features[9:18](x), self.encoder.features[18:27](x), self.encoder.features[27:36](x))

class DOWN(nn.Module):
    def __init__(self):
        super(DOWN, self).__init__()
        self.down = nn.MaxPool2d(2)
    def forward(self, x):
        x2 = self.down(x)
        return x, x2, self.down(x2), self.down(self.down(x2))

class GT_Model(nn.Module):
    def __init__(self, encoder_I, encoder_R):
        super(GT_Model, self).__init__()
        self.Encoder_I = Encoder(encoder_I)
        self.Encoder_R = Encoder(encoder_R)
        self.upy1 = up(1024, 256, 4, 512)  
        self.upy2 = up(512, 128, 4, 256)
        self.upy3 = up(256, 64, 2, 128)
        self.upy4 = up(128, 64, 2, 64) 
        self.outTc = nn.Conv2d(64, 3, 1)
        self.down = DOWN()
           
    def forward(self, input_I, input_R, map_thr=None, map_encoder=None):
        I_x1, I_x2, I_x3, I_x4, I_x5 = self.Encoder_I(input_I)
        R_x1, R_x2, R_x3, R_x4, R_x5 = self.Encoder_R(input_R)
             
        y4, mask4, m4_e, m4_d = self.upy1(I_x5, I_x4, R_x4)
        y3, mask3, m3_e, m3_d = self.upy2(y4, I_x3, R_x3) 
        y2, mask2, m2_e, m2_d = self.upy3(y3, I_x2, R_x2)
        y1, mask1, m1_e, m1_d = self.upy4(y2, I_x1, R_x1)
        y = self.outTc(y1)
        
        if not self.training:
            return F.sigmoid(y),
        
        m_1, m_2, m_3, m_4 = self.down(map_thr)
        m_1e, m_2e, m_3e, m_4e = self.down(map_encoder)
        return F.sigmoid(y), [m_1, m_2, m_3, m_4], [R_x1, R_x2, R_x3, R_x4], \
               [mask1, mask2, mask3, mask4], [m1_e, m2_e, m3_e, m4_e], [m_1e, m_2e, m_3e, m_4e]

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, kernel_size=4, stride=1, padding=1), nn.Sigmoid()
        )
    def forward(self, blended, transmission):
        return self.layers(torch.cat([blended, transmission], dim=1))