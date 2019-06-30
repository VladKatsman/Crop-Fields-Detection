import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

__all__ = ['DenseNet', 'densenet161']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

encoder_params = {'densenet161':
                                {'filters': [96, 384, 768, 2112, 2208],
                                 'decoder_filters': [64, 128, 256, 256],
                                 'last_upsample': 64,
                                 'url': None}
}


def densenet161(pretrained=True, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but previous _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet161'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.state_dict()['features.conv0.weight'][:, :3, ...] = state_dict['features.conv0.weight'].data

        pretrained_dict = {k: v for k, v in state_dict.items() if k != 'features.conv0.weight'}
        model.load_state_dict(pretrained_dict, strict=False)

    return model


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)), # 4<3
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out


class BasicConvAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1, activation=nn.ReLU, bias=True):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=bias)
        self.use_act = activation is not None
        if self.use_act:
            self.act = activation()

    def forward(self, x):
        x = self.op(x)
        if self.use_act:
            x = self.act(x)
        return x


class Conv1x1(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size=1, dilation=dilation, activation=None, bias=bias)


class Conv3x3(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size=3, dilation=dilation, activation=None)


class ConvReLu1x1(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size=1, dilation=dilation, activation=nn.ReLU)


class ConvReLu3x3(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size=3, dilation=dilation, activation=nn.ReLU)


class BasicUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=nn.ReLU, mode='nearest'):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * 1
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=1)
        self.use_act = activation is not None
        self.mode = mode
        if self.use_act:
            self.act = activation()

    def forward(self, x):
        x = F.upsample(x, scale_factor=2, mode=self.mode)
        x = self.op(x)
        if self.use_act:
            x = self.act(x)
        return x


class AbstractModel(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def initialize_encoder(self, model, model_url, num_channels_changed=False):
        if os.path.isfile(model_url):
            pretrained_dict = torch.load(model_url)
        else:
            pretrained_dict = model_zoo.load_url(model_url)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
            pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if num_channels_changed:
            model.state_dict()[self.first_layer_params_name + '.weight'][:, :3, ...] = pretrained_dict[self.first_layer_params_name + '.weight'].data
            skip_layers = [self.first_layer_params_name, self.first_layer_params_name + ".weight"]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               not any(k.startswith(s) for s in skip_layers)}
        model.load_state_dict(pretrained_dict, strict=False)

    @property
    def first_layer_params_name(self):
        return 'conv1'


class EncoderDecoder(AbstractModel):
    def __init__(self, num_classes, num_channels=3, encoder_name='resnet34'):
        if not hasattr(self, 'first_layer_stride_two'):
            self.first_layer_stride_two = False
        if not hasattr(self, 'decoder_block'):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck

        self.filters = encoder_params[encoder_name]['filters']
        self.decoder_filters = encoder_params[encoder_name].get('decoder_filters', self.filters[:-1])
        self.last_upsample_filters = encoder_params[encoder_name].get('last_upsample', self.decoder_filters[0]//2)

        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        self.bottlenecks = nn.ModuleList([self.bottleneck_type(self.filters[-i - 2] + f, f) for i, f in
                                          enumerate(reversed(self.decoder_filters[:]))])

        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(0, len(self.decoder_filters))])

        if self.first_layer_stride_two:
            self.last_upsample = self.decoder_block(self.decoder_filters[0], self.last_upsample_filters,
                                                    self.last_upsample_filters)

        self.final = self.make_final_classifier(
            self.last_upsample_filters if self.first_layer_stride_two else self.decoder_filters[0], num_classes)

        self._initialize_weights()

        encoder = densenet161()
        self.encoder_stages = nn.ModuleList([self.get_encoder(encoder, idx) for idx in range(len(self.filters))])
        if encoder_params[encoder_name]['url'] is not None:
            self.initialize_encoder(encoder, encoder_params[encoder_name]['url'], num_channels != 3)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        enc_results = []
        for stage in self.encoder_stages:
            x = stage(x)
            enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())
        last_dec_out = enc_results[-1]
        x = last_dec_out
        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        if self.first_layer_stride_two:
            x = self.last_upsample(x)

        f = self.final(x)

        return f

    def get_decoder(self, layer):
        in_channels = self.filters[layer + 1] if layer + 1 == len(self.decoder_filters) else self.decoder_filters[
            layer + 1]
        return self.decoder_block(in_channels, self.decoder_filters[layer], self.decoder_filters[max(layer, 0)])

    def make_final_classifier(self, in_filters, num_classes):
        return nn.Sequential(
            nn.Conv2d(in_filters, num_classes, 1, padding=0)
        )

    def get_encoder(self, encoder, layer):
        raise NotImplementedError

    @property
    def first_layer_params(self):
        return _get_layers_params([self.encoder_stages[0]])

    @property
    def layers_except_first_params(self):
        layers = get_slice(self.encoder_stages, 1, -1) + [self.bottlenecks, self.decoder_stages, self.final]
        return _get_layers_params(layers)


def _get_layers_params(layers):
    return sum((list(l.parameters()) for l in layers), [])


def get_slice(features, start, end):
    if end == -1:
        end = len(features)
    return [features[i] for i in range(start, end)]


class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest') # changed to nearest
        return self.layer(x)


class DensenetUnet(EncoderDecoder):
    def __init__(self, seg_classes, backbone_arch='densenet161'):
        self.first_layer_stride_two = True
        super().__init__(seg_classes, 3, backbone_arch)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.features.conv0,  # conv
                encoder.features.norm0,  # bn
                encoder.features.relu0  # relu
            )
        elif layer == 1:
            return nn.Sequential(encoder.features.pool0, encoder.features.denseblock1)
        elif layer == 2:
            return nn.Sequential(encoder.features.transition1, encoder.features.denseblock2)
        elif layer == 3:
            return nn.Sequential(encoder.features.transition2, encoder.features.denseblock3)
        elif layer == 4:
            return nn.Sequential(encoder.features.transition3, encoder.features.denseblock4, encoder.features.norm5, nn.ReLU())