import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, momentum=0.1, 
                 radius=False):
        super(ConvBnRelu, self).__init__()

        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                padding=padding, stride=stride)

        self.bn = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class StackEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, padding, momentum=0.5, radius=False):
        super(StackEncoder, self).__init__()
        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3),
                                 stride=1, padding=padding, momentum=momentum, radius=radius)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3),
                                 stride=1, padding=padding, momentum=momentum)
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        x = self.convr1(x)
        x = self.convr2(x)
        x_trace = x
        x = self.maxPool(x)
        return x, x_trace

class StackDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, padding, momentum=0.5, radius=False):
        super(StackDecoder, self).__init__()

        # self.upSample = nn.Upsample(size=upsample_size, scale_factor=(2,2), mode='bilinear')
        self.upSample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2)

        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding,
                                 momentum=momentum, radius=radius)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding,
                                 momentum=momentum)

    def _crop_concat(self, upsampled, bypass):

        margin = bypass.size()[2] - upsampled.size()[2]
        c = margin // 2
        if margin % 2 == 1:
            bypass = F.pad(bypass, (-c, -c - 1, -c, -c - 1))
        else:
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x, down_tensor):
        # x = self.upSample(x)
        x = self.transpose_conv(x)
        x = self._crop_concat(x, down_tensor)
        x = self.convr1(x)
        x = self.convr2(x)
        return x

class Unet2D(nn.Module):
    
    def __init__(self, in_shape, padding, momentum, num_classes):
        super(Unet2D, self).__init__()
        channels, heights, width = in_shape
        self.padding = padding

        self.down1 = StackEncoder(channels, 64, padding, momentum=momentum)
        self.down2 = StackEncoder(64, 128, padding, momentum=momentum)
        self.down3 = StackEncoder(128, 256, padding, momentum=momentum)
        self.down4 = StackEncoder(256, 512, padding, momentum=momentum)

        self.center1 = ConvBnRelu(512, 1024, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum)
        self.center2 = ConvBnRelu(1024, 1024, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum)

        self.up1 = StackDecoder(in_channels=1024, out_channels=512, padding=padding, momentum=momentum)
        self.up2 = StackDecoder(in_channels=512, out_channels=256, padding=padding, momentum=momentum)
        self.up3 = StackDecoder(in_channels=256, out_channels=128, padding=padding, momentum=momentum)
        self.up4 = StackDecoder(in_channels=128, out_channels=64, padding=padding, momentum=momentum)

        self.output_seg_map = nn.Conv2d(64, num_classes, kernel_size=(1, 1), padding=0, stride=1)
        self.output_up_seg_map = nn.Upsample(size=(heights, width), mode='nearest')

    def forward(self, x):
        x, x_trace1 = self.down1(x)
        x, x_trace2 = self.down2(x)
        x, x_trace3 = self.down3(x)
        x, x_trace4 = self.down4(x)

        x_bottom_block = self.center1(x)
        x = self.center2(x_bottom_block)
        x = self.up1(x, x_trace4)
        x = self.up2(x, x_trace3)
        x = self.up3(x, x_trace2)
        x = self.up4(x, x_trace1)

        out = self.output_seg_map(x)

        # need size down output
        if self.padding == 0:
            out = self.output_up_seg_map(out)

        return out

if __name__ == '__main__':
    pass