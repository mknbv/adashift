from torch import nn
import torch.nn.init as nninit

def avg_pool2d(x):
    '''Twice differentiable implementation of 2x2 average pooling.'''
    return (x[:, :, ::2, ::2] + x[:, :, 1::2, ::2] + x[:, :, ::2, 1::2] + x[:, :, 1::2, 1::2]) / 4

class GeneratorBlock(nn.Module):
    '''ResNet-style block for the generator model.'''

    def __init__(self, in_chans, out_chans, upsample=False):
        super().__init__()

        self.upsample = upsample

        if in_chans != out_chans:
            self.shortcut_conv = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        else:
            self.shortcut_conv = None
        self.bn1 = nn.BatchNorm2d(in_chans)
        self.conv1 = nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_chans)
        self.conv2 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)

    def forward(self, *inputs):
        x = inputs[0]

        if self.upsample:
            shortcut = nn.functional.upsample(x, scale_factor=2, mode='nearest')
        else:
            shortcut = x

        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)

        x = self.bn1(x)
        x = nn.functional.relu(x, inplace=False)
        if self.upsample:
            x = nn.functional.upsample(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.bn2(x)
        x = nn.functional.relu(x, inplace=False)
        x = self.conv2(x)

        return x + shortcut

class Generator(nn.Module):
    '''The generator model.'''

    def __init__(self, feats=128):
        super().__init__()

        self.feats = feats
        self.input_linear = nn.Linear(feats, 4*4*feats)
        self.block1 = GeneratorBlock(feats, feats, upsample=True)
        self.block2 = GeneratorBlock(feats, feats, upsample=True)
        self.block3 = GeneratorBlock(feats, feats, upsample=True)
        self.output_bn = nn.BatchNorm2d(feats)
        self.output_conv = nn.Conv2d(feats, 3, kernel_size=3, padding=1)

        # Apply Xavier initialisation to the weights
        relu_gain = nninit.calculate_gain('relu')
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                gain = relu_gain if module != self.input_linear else 1.0
                nninit.xavier_uniform(module.weight.data, gain=gain)
                module.bias.data.zero_()

        self.last_output = None

    def forward(self, *inputs):
        x = inputs[0]

        x = self.input_linear(x)
        x = x.view(-1, self.feats, 4, 4)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.output_bn(x)
        x = nn.functional.relu(x, inplace=False)
        x = self.output_conv(x)
        x = nn.functional.tanh(x)

        self.last_output = x

        return x

class DiscriminatorBlock(nn.Module):
    '''ResNet-style block for the discriminator model.'''

    def __init__(self, in_chans, out_chans, downsample=False, first=False):
        super().__init__()

        self.downsample = downsample
        self.first = first

        if in_chans != out_chans:
            self.shortcut_conv = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        else:
            self.shortcut_conv = None
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)

    def forward(self, *inputs):
        x = inputs[0]

        if self.downsample:
            shortcut = avg_pool2d(x)
        else:
            shortcut = x

        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)

        if not self.first:
            x = nn.functional.relu(x, inplace=False)
        x = self.conv1(x)
        x = nn.functional.relu(x, inplace=False)
        x = self.conv2(x)
        if self.downsample:
            x = avg_pool2d(x)

        return x + shortcut

class Discriminator(nn.Module):
    '''The discriminator (aka critic) model.'''

    def __init__(self, feats):
        super().__init__()

        self.feats = feats
        self.block1 = DiscriminatorBlock(3, feats, downsample=True, first=True)
        self.block2 = DiscriminatorBlock(feats, feats, downsample=True)
        self.block3 = DiscriminatorBlock(feats, feats, downsample=False)
        self.block4 = DiscriminatorBlock(feats, feats, downsample=False)
        self.output_linear = nn.Linear(self.feats, 1)

        # Apply Xavier initialisation to the weights
        relu_gain = nninit.calculate_gain('relu')
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                gain = relu_gain if module != self.block1.conv1 else 1.0
                nninit.xavier_uniform(module.weight.data, gain=gain)
                module.bias.data.zero_()

    def forward(self, *inputs):
        x = inputs[0]

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = nn.functional.relu(x, inplace=False)
        x = x.mean(-1, keepdim=False).mean(-1, keepdim=False)
        x = x.view(-1, self.feats)
        x = self.output_linear(x)

        return x
