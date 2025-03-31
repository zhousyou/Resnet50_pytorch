import torch
import torch.nn as nn
from torchsummary import summary


def conv3x3(in_channels, out_channels, stride = 1, padding = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)

def conv1x1(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

class Basicblock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super().__init__()
        self.conv1 = conv3x3(in_channels=in_channels, out_channels=out_channels, stride=stride,padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(in_channels=out_channels, out_channels= out_channels, stride=stride, padding=1)
        self.stride = stride
        self.dowmsample = downsample
    
    def forward(self, x):

        identity = x
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)

        if self.dowmsample is not None:
            identity = self.dowmsample(identity)

        out += identity
        out = self.relu(out)
        return out

class Bottleblock(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super().__init__()
        self.conv1 = conv1x1(in_channels=in_channels, out_channels=out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(in_channels=out_channels, out_channels=out_channels,stride=stride, padding=1)
        self.conv3 = conv1x1(in_channels=out_channels, out_channels=out_channels*self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out


        
class Resnet(nn.Module):
    def __init__(self, block, layers, num_classes = 100):
        super().__init__()
        # self.zeropad = nn.ZeroPad2d(padding=3)
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,
                               stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, out_channels, blocks, stride = 1):
        downsample = None
        if stride!=1 or self.in_channels != out_channels*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, 
                          out_channels= out_channels * block.expansion, 
                          kernel_size=1),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride = stride, downsample=downsample))
        self.in_channels = out_channels*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
    
# model = Resnet(Bottleblock, [3,4,6,3])
# summary(model,(3,224,224))
model = Bottleblock(256, 128)
summary(model, (256, 56,56))
