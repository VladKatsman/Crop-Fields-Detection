import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

pretrained_models = {

    'faces': '/data/dev/projects/face_tran_experiments/results/resunet_faces/model_best.pth'
}
def make_conv_block(in_nc, out_nc, kernel_size=3, stride=1, padding=None, bias=False,
                     padding_type='zero', norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(True), use_dropout=False):
    conv_block = []
    p = 0
    if padding_type is not None:
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(kernel_size // 2 if padding is None else padding)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(kernel_size // 2 if padding is None else padding)]
        elif padding_type == 'zero':
            p = kernel_size // 2 if padding is None else padding
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
    elif padding is not None:
        p = padding

    conv_block.append(nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=p, bias=bias))
    if norm_layer is not None:
        conv_block.append(norm_layer(out_nc))
    if act_layer is not None:
        conv_block.append(act_layer)

    if use_dropout:
        conv_block += [nn.Dropout(0.5)]

    return conv_block


class DownBlock(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size=3, padding_type='zero',
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(True)):
        super(DownBlock, self).__init__()
        model = make_conv_block(in_nc, out_nc, kernel_size, 2, padding_type=padding_type,
                                norm_layer=norm_layer, act_layer=act_layer)
        model += make_conv_block(out_nc, out_nc, kernel_size, 1, padding_type=padding_type,
                                 norm_layer=norm_layer, act_layer=act_layer)
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class UpBlock(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size=3, padding_type='zero',
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(True)):
        super(UpBlock, self).__init__()
        # model = [nn.Upsample(scale_factor=2, mode='bilinear')]
        model = make_conv_block(in_nc, out_nc, kernel_size, 1, padding_type=padding_type,
                                 norm_layer=norm_layer, act_layer=act_layer)
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.model(x)


class ResnetBlock(nn.Module):
    def __init__(self, planes, kernel_size=3, expansion=1, padding_type='zero',
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        model = make_conv_block(planes, planes*expansion, kernel_size, padding_type=padding_type,
                                norm_layer=norm_layer, act_layer=act_layer, use_dropout=use_dropout)
        model += make_conv_block(planes*expansion, planes, kernel_size, padding_type=padding_type,
                                 norm_layer=norm_layer, act_layer=None, use_dropout=False)
        self.model = nn.Sequential(*model)
        self.act = act_layer

    def forward(self, x):
        out = x + self.model(x)
        out = self.act(out)
        return out


class FlatBlock(nn.Module):
    def __init__(self, planes, kernel_size=3, layers=1, padding_type='zero',
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(True)):
        super(FlatBlock, self).__init__()
        if layers <= 0:
            self.model = None
        else:
            model = []
            for i in range(layers):
                model.append(ResnetBlock(planes, kernel_size, 1, padding_type, norm_layer, act_layer))
            self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.model is None:
            return x
        return self.model(x)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|-- flat --|
class SkipConnectionBlock(nn.Module):
    def __init__(self, ngf, sub_ngf, down_block=None, submodule=None, up_block=None, flat_block=None, flat_layers=1,
                 padding_type='zero', norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(inplace=True), use_dropout=False):
        super(SkipConnectionBlock, self).__init__()
        self.submodule = submodule
        if submodule is not None:
            assert down_block is not None and up_block is not None
            self.down_block = down_block(ngf, sub_ngf, 3, padding_type, norm_layer, act_layer)
            self.up_block = up_block(sub_ngf, ngf, 3, padding_type, norm_layer, act_layer)
        if flat_block is not None:
            self.flat_block = flat_block(ngf, 3, flat_layers, padding_type, norm_layer, act_layer)
        else:
            self.flat_block = None

    def forward(self, x):
        if self.submodule is not None:
            x = x + self.up_block(self.submodule(self.down_block(x)))
        if self.flat_block is not None:
            return self.flat_block(x)
        return x


class ResUNet(nn.Module):
    def __init__(self, down_block=DownBlock, up_block=UpBlock, flat_block=FlatBlock, in_nc=3, out_nc=3, ngf=64,
                 flat_layers=(0, 0, 0, 3), norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(inplace=True), use_dropout=False):
        super(ResUNet, self).__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.in_conv = nn.Sequential(*make_conv_block(in_nc, ngf, kernel_size=3, norm_layer=norm_layer,
                                                      act_layer=act_layer, use_dropout=use_dropout))
        self.out_conv = make_conv_block(ngf, out_nc, kernel_size=3, norm_layer=None, act_layer=None)
        self.out_conv.append(nn.Tanh())
        self.out_conv = nn.Sequential(*self.out_conv)

        self.levels = len(flat_layers)
        unet_block = None
        for i in range(1, self.levels + 1):
            unet_block = SkipConnectionBlock(ngf * (2 ** (self.levels - i)), ngf * (2 ** (self.levels - i + 1)),
                                             down_block, unet_block, up_block, flat_block, flat_layers=flat_layers[-i],
                                             norm_layer=norm_layer, act_layer=act_layer, use_dropout=use_dropout)
        self.inner = unet_block

    def forward(self, x):
        x = self.in_conv(x)
        x = self.inner(x)
        x = self.out_conv(x)

        return x


def faces_net():
    path = pretrained_models['faces']
    model = ResUNet(out_nc=2, flat_layers=(2, 2, 2, 3))
    checkpoint = torch.load(path)
    weights = checkpoint['state_dict']
    model.load_state_dict(weights)

    return model



def main(model='res_unet.ResUNet', res=256):
    from face_tran.utils.obj_factory import obj_factory
    model = obj_factory(model)
    img = torch.rand(1, model.in_nc, res, res)
    pred = model(img)
    print(pred.shape)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('res_unet test')
    parser.add_argument('model', default='res_unet.ResUNet',
                        help='model object')
    parser.add_argument('-r', '--res', default=256, type=int,
                        metavar='N', help='image resolution')
    args = parser.parse_args()
    main(args.model, args.res)
