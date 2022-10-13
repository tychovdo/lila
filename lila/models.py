import math
import torch
import numpy as np
from torch import nn

from asdfghjkl.operations import Bias, Scale
from asdfghjkl.operations.conv_aug import Conv2dAug


def get_activation(act_str):
    if act_str == 'relu':
        return nn.ReLU
    elif act_str == 'tanh':
        return nn.Tanh
    elif act_str == 'selu':
        return nn.SELU
    elif act_str == 'silu':
        return nn.SiLU
    else:
        raise ValueError('invalid activation')


class MaxPool2dAug(nn.MaxPool2d):
    
    def forward(self, input):
        k_aug = input.shape[1]
        input = super().forward(input.flatten(start_dim=0, end_dim=1))
        return input.reshape(-1, k_aug, *input.shape[1:])

        
class AvgPool2dAug(nn.AvgPool2d):
    
    def forward(self, input):
        k_aug = input.shape[1]
        input = super().forward(input.flatten(start_dim=0, end_dim=1))
        return input.reshape(-1, k_aug, *input.shape[1:])
    
    
class AdaptiveAvgPool2dAug(nn.AdaptiveAvgPool2d):

    def forward(self, input):
        k_aug = input.shape[1]
        input = super().forward(input.flatten(start_dim=0, end_dim=1))
        return input.reshape(-1, k_aug, *input.shape[1:])

        
class MLP(nn.Sequential):
    def __init__(self, input_size, width, depth, output_size, activation='relu', 
                 bias=True, fixup=False, augmented=True):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.width = width
        self.depth = depth
        hidden_sizes = depth * [width]
        self.activation = activation
        if output_size is not None:
            self.output_size = output_size
        else:
            self.output_size = 1
        flatten_start_dim = 2 if augmented else 1
        act = get_activation(activation)

        self.add_module('flatten', nn.Flatten(start_dim=flatten_start_dim))

        if len(hidden_sizes) == 0:  # i.e. when depth == 0.
            # Linear Model
            self.add_module('lin_layer', nn.Linear(self.input_size, self.output_size, bias=bias))
        else:
            # MLP
            in_outs = zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)
            for i, (in_size, out_size) in enumerate(in_outs):
                self.add_module(f'layer{i+1}', nn.Linear(in_size, out_size, bias=bias))
                if fixup:
                    self.add_module(f'bias{i+1}b', Bias())
                    self.add_module(f'scale{i+1}b', Scale())
                self.add_module(f'{activation}{i+1}', act())
            self.add_module('out_layer', nn.Linear(hidden_sizes[-1], self.output_size, bias=bias))


class LeNet(nn.Sequential):
    
    def __init__(self, in_channels=1, n_out=10, activation='relu', n_pixels=28):
        super().__init__()
        mid_kernel_size = 3 if n_pixels == 28 else 5
        act = get_activation(activation)
        self.add_module('conv1', Conv2dAug(in_channels, 6, 5, 1))
        self.add_module('act1', act())
        self.add_module('pool1', MaxPool2dAug(2)) 
        self.add_module('conv2', Conv2dAug(6, 16, mid_kernel_size, 1))
        self.add_module('act2', act())
        self.add_module('pool2', MaxPool2dAug(2))
        self.add_module('conv3', Conv2dAug(16, 120, 5, 1))
        self.add_module('flatten', nn.Flatten(start_dim=2, end_dim=4))
        self.add_module('act3', act())
        self.add_module('lin1', torch.nn.Linear(120*1*1, 84))
        self.add_module('act4', act())
        self.add_module('linout', torch.nn.Linear(84, n_out))

            
def conv3x3(in_planes, out_planes, stride=1, augmented=True):
    """3x3 convolution with padding"""
    Conv2d = Conv2dAug if augmented else nn.Conv2d
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                  padding=1, bias=False)


class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, augmented=True):
        super(FixupBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.augmented = augmented
        self.bias1a = Bias()
        self.conv1 = conv3x3(inplanes, planes, stride, augmented=augmented)
        self.bias1b = Bias()
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = Bias()
        self.conv2 = conv3x3(planes, planes, augmented=augmented)
        self.scale = Scale()
        self.bias2b = Bias()
        self.downsample = downsample

    def forward(self, x):
        identity = x

        biased_x = self.bias1a(x)
        out = self.conv1(biased_x)
        out = self.relu(self.bias1b(out))

        out = self.conv2(self.bias2a(out))
        out = self.bias2b(self.scale(out))

        if self.downsample is not None:
            identity = self.downsample(biased_x)
            cat_dim = 2 if self.augmented else 1
            identity = torch.cat((identity, torch.zeros_like(identity)), cat_dim)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    FixupResnet-depth where depth is a `3 * 2 * n + 2` with `n` blocks per residual layer.
    The two added layers are the input convolution and fully connected output.
    """

    def __init__(self, depth, num_classes=10, in_planes=16, in_channels=3, augmented=True):
        super(ResNet, self).__init__()
        self.output_size = num_classes
        assert (depth - 2) % 6 == 0, 'Invalid ResNet depth, has to conform to 6 * n + 2'
        layer_size = (depth - 2) // 6
        layers = 3 * [layer_size]
        self.num_layers = 3 * layer_size
        self.inplanes = in_planes
        self.augmented = augmented
        AdaptiveAvgPool2d = AdaptiveAvgPool2dAug if augmented else nn.AdaptiveAvgPool2d
        self.conv1 = conv3x3(in_channels, in_planes, augmented=augmented)
        self.bias1 = Bias()
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(FixupBasicBlock, in_planes, layers[0])
        self.layer2 = self._make_layer(FixupBasicBlock, in_planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(FixupBasicBlock, in_planes * 4, layers[2], stride=2)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=2 if augmented else 1)
        self.bias2 = Bias()
        self.fc = nn.Linear(in_planes * 4, num_classes)

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.weight, 
                                mean=0, 
                                std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        AvgPool2d = AvgPool2dAug if self.augmented else nn.AvgPool2d
        if stride != 1:
            downsample = AvgPool2d(1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, augmented=self.augmented))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes, augmented=self.augmented))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bias1(x))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(self.bias2(x))

        return x

        
##########################################################################################
# Wide ResNet (for WRN16-4)
##########################################################################################
# Adapted from https://github.com/hendrycks/outlier-exposure/blob/master/CIFAR/models/wrn.py
class WRNFixupBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, augmented=False):
        super(WRNFixupBasicBlock, self).__init__()
        self.bias1 = Bias()
        self.relu1 = nn.ReLU(inplace=True)
        basemodul = Conv2dAug if augmented else nn.Conv2d 
        self.augmented = augmented
        self.conv1 = basemodul(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bias2 = Bias()
        self.relu2 = nn.ReLU(inplace=True)
        self.bias3 = Bias()
        self.conv2 = basemodul(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bias4 = Bias()
        self.scale1 = Scale()
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and basemodul(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bias1(x))
        else:
            out = self.relu1(self.bias1(x))
        if self.equalInOut:
            out = self.bias3(self.relu2(self.bias2(self.conv1(out))))
        else:
            out = self.bias3(self.relu2(self.bias2(self.conv1(x))))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.bias4(self.scale1(self.conv2(out)))
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class WRNFixupNetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, augmented=False):
        super(WRNFixupNetworkBlock, self).__init__()
        self.augmented = augmented
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, self.augmented))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=16, widen_factor=4, num_classes=10, dropRate=0.0, augmented=True):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = WRNFixupBasicBlock
        # 1st conv before any network block
        self.num_layers = n * 3
        basemodul = Conv2dAug if augmented else nn.Conv2d 
        self.augmented = augmented
        self.conv1 = basemodul(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bias1 = Bias()
        # 1st block
        self.block1 = WRNFixupNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, augmented=augmented)
        # 2nd block
        self.block2 = WRNFixupNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, augmented=augmented)
        # 3rd block
        self.block3 = WRNFixupNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, augmented=augmented)
        # global average pooling and classifier
        self.bias2 = Bias()
        self.relu = nn.ReLU()
        self.pool = AvgPool2dAug(8) if augmented else nn.AvgPool2d(8)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, WRNFixupBasicBlock):
                conv = m.conv1
                k = conv.weight.shape[0] * np.prod(conv.weight.shape[2:])
                nn.init.normal_(conv.weight,
                                mean=0,
                                std=np.sqrt(2. / k) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
                if m.convShortcut is not None:
                    cs = m.convShortcut
                    k = cs.weight.shape[0] * np.prod(cs.weight.shape[2:])
                    nn.init.normal_(cs.weight,
                                    mean=0,
                                    std=np.sqrt(2. / k))
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.bias1(self.conv1(x))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(out)
        out = self.pool(out)
        if self.augmented:
            out = out.flatten(start_dim=2)
        else:
            out = out.flatten(start_dim=1)
        return self.fc(self.bias2(out))


if __name__ == '__main__':
    # Example data
    n_data, n_aug, n_channels, n_pixels = 32, 11, 3, 32
    n_outputs = 10
    device = 'cpu'
    output_shape = torch.Size([n_data, n_outputs])
    output_shape_aug = torch.Size([n_data, n_aug, n_outputs])
    X = torch.randn(n_data, n_channels, n_pixels, n_pixels, device=device)
    X_aug = torch.randn(n_data, n_aug, n_channels, n_pixels, n_pixels, device=device)

    # 1. Test MLP
    model = MLP(n_channels * n_pixels ** 2, 100, 3, n_outputs, augmented=False).to(device)
    assert model(X).size() == output_shape
    model_aug = MLP(n_channels * n_pixels ** 2, 100, 3, n_outputs, augmented=True).to(device)
    assert model_aug(X_aug).size() == output_shape_aug
    print('MLP passed test.')

    # 2. Test CNN
    model = LeNet(in_channels=n_channels, n_pixels=n_pixels).to(device)
    assert model_aug(X_aug).size() == output_shape_aug
    print('LeNet CNN passed test')

    # 3. Test ResNet-14 
    model = ResNet(14, n_outputs, in_channels=n_channels, augmented=False).to(device)
    assert model(X).size() == output_shape
    model = ResNet(14, n_outputs, in_channels=n_channels, augmented=True).to(device)
    assert model_aug(X_aug).size() == output_shape_aug
    print('ResNet passed test')

    # 4. Test WRN
    model = WideResNet(16, 4, augmented=False).to(device)
    assert model(X).size() == output_shape
    model = WideResNet(16, 4, augmented=True).to(device)
    assert model(X_aug).size() == output_shape_aug
    print('WRN passed test')
