import torch
import torch.nn as nn
import torch.nn.functional as F
import uuid

# Convolutional Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
# Encoder and Decoder Blocks
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()

        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, type='normal'):
        super(DecoderBlock, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        if type == 'normal':
            self.conv = ConvBlock(in_channels, out_channels)
        elif type == 'attention':
            self.conv = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x, p):
        x = self.up(x)
        h_diff = p.size()[2] - x.size()[2]
        w_diff = p.size()[3] - x.size()[3]

        if h_diff > 0:
            x = F.pad(x, (0, 0, 0, h_diff))
        elif h_diff < 0:
            p = F.pad(p, (0, 0, 0, -h_diff))

        if w_diff > 0:
            x = F.pad(x, (0, w_diff, 0, 0))
        elif w_diff < 0:
            p = F.pad(p, (0, -w_diff, 0, 0))
        
        x = torch.cat([x, p], dim=1)
        return self.conv(x)
    
# Attention Gate
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        g1 = F.interpolate(g1, size=x.size()[2:], mode='bilinear', align_corners=True)

        x1 = self.W_x(x)
        
        psi = F.relu(g1 + x1, inplace=True)
        psi = self.psi(psi)
        return x * psi

# UNET Model
class UNET(nn.Module):
    def __init__(self, num_classes):
        super(UNET, self).__init__()
        self.uid = uuid.uuid4().hex[:8]

        self.encoder1 = EncoderBlock(3, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        self.center = ConvBlock(512, 1024)

        self.decoder4 = DecoderBlock(1024, 512)
        self.decoder3 = DecoderBlock(512, 256)
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder1 = DecoderBlock(128, 64)

        self.output = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1, p1 = self.encoder1(x)
        e2, p2 = self.encoder2(p1)
        e3, p3 = self.encoder3(p2)
        e4, p4 = self.encoder4(p3)

        c = self.center(p4)

        d4 = self.decoder4(c, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)

        return self.output(d1)
    
# UNET Model with Attention
class UNETWithAttention(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(UNETWithAttention, self).__init__()
        self.NUM_CLASSES = NUM_CLASSES
        self.uid = uuid.uuid4().hex[:8]

        self.encoder1 = EncoderBlock(3, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        self.middle = ConvBlock(512, 1024)
        
        self.attention4 = AttentionGate(1024, 512, 256)
        self.decoder4 = DecoderBlock(1024, 512, 'attention')
        self.attention3 = AttentionGate(512, 256, 128)
        self.decoder3 = DecoderBlock(512, 256, 'attention')
        self.attention2 = AttentionGate(256, 128, 64)
        self.decoder2 = DecoderBlock(256, 128, 'attention')
        self.attention1 = AttentionGate(128, 64, 32)
        self.decoder1 = DecoderBlock(128, 64, 'attention')

        self.final = nn.Conv2d(64, self.NUM_CLASSES, kernel_size=1)

    def forward(self, x):
        e1, p1 = self.encoder1(x)
        e2, p2 = self.encoder2(p1)
        e3, p3 = self.encoder3(p2)
        e4, p4 = self.encoder4(p3)

        c = self.middle(p4)

        d4 = self.decoder4(c, self.attention4(c, e4))
        d3 = self.decoder3(d4, self.attention3(d4, e3))
        d2 = self.decoder2(d3, self.attention2(d3, e2))
        d1 = self.decoder1(d2, self.attention1(d2, e1))

        return self.final(d1)