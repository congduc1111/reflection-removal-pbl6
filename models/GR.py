import torch
import torch.nn as nn
import torch.nn.functional as F

class up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_num):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        conv_block = double_conv if conv_num == 2 else forth_conv
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        return self.conv(torch.cat([x2, x1], dim=1))

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.BatchNorm2d(in_ch, track_running_stats=False), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch, track_running_stats=False), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class forth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(forth_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.BatchNorm2d(in_ch, track_running_stats=False), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.BatchNorm2d(in_ch, track_running_stats=False), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.BatchNorm2d(in_ch, track_running_stats=False), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch, track_running_stats=False), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class GR_Model(nn.Module):
    def __init__(self, encoder):
        super(GR_Model, self).__init__()
        self.encoder = encoder
        self.upz1 = up(1024, 256, 4)  
        self.upz2 = up(512, 128, 4)
        self.upz3 = up(256, 64, 2)
        self.upz4 = up(128, 64, 2)  
        self.outzc = nn.Conv2d(64, 3, 1)        
        
    def forward(self, x):
        x1 = self.encoder.features[0:4](x)
        x2 = self.encoder.features[4:9](x)
        x3 = self.encoder.features[9:18](x)
        x4 = self.encoder.features[18:27](x)
        x5 = self.encoder.features[27:36](x)

        z4 = self.upz1(x5, x4)   
        z3 = self.upz2(z4, x3)   
        z2 = self.upz3(z3, x2)
        z1 = self.upz4(z2, x1)
        return F.sigmoid(self.outzc(z1))