import os
import importlib


known_modules = {
    # criterions
    'ssim': 'field_boundaries.criterions.ssim',

    # data
    'ricult_dataset': 'field_boundaries.data.ricult_dataset',

    # models
    'unet': 'field_boundaries.models.unet',  # added by Vlad
    'unet_res': 'field_boundaries.models.unet_res',  # added by Vlad (Yuval implementation)
    'res_unet_split': 'field_boundaries.models.res_unet_split',
    'densenet_unet': 'field_boundaries.models.densenet_unet',

    # utils
    # added by Vlad for segmentation
    'seg_transforms': 'field_boundaries.utils.seg_transforms',
    'losses': 'field_boundaries.utils.utils.losses',
    'schedulers': 'field_boundaries.utils.utils.schedulers',
    'landmark_transforms': 'field_boundaries.utils.utils.landmark_transforms',
    'optimizers': 'field_boundaries.utils.optimizers',

    # Torch
    'nn': 'torch.nn',
    'optim': 'torch.optim',
    'lr_scheduler': 'torch.optim.lr_scheduler',


    # Torchvision
    'datasets': 'torchvision.datasets',
    'transforms': 'torchvision.transforms'
}

known_classes = {
}


def extract_args(*args, **kwargs):
    return args, kwargs


def obj_factory(obj_exp, *args, **kwargs):
    if not isinstance(obj_exp, str):
        return obj_exp

    # Handle arguments
    if '(' in obj_exp and ')' in obj_exp:
        args_exp = obj_exp[obj_exp.find('('):]
        obj_args, obj_kwargs = eval('extract_args' + args_exp)

        # Concatenate arguments
        args = obj_args + args
        kwargs.update(obj_kwargs)

        obj_exp = obj_exp[:obj_exp.find('(')]

    # From here we can assume that dots in the remaining of the expression
    # only separate between modules and classes
    module_name, class_name = os.path.splitext(obj_exp)
    class_name = class_name[1:]
    module = importlib.import_module(known_modules[module_name] if module_name in known_modules else module_name)
    module_class = getattr(module, class_name)
    class_instance = module_class(*args, **kwargs)

    return class_instance


def main(obj_exp):
    obj = obj_factory(obj_exp)
    print(obj)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('utils test')
    parser.add_argument('obj_exp', help='object string')
    args = parser.parse_args()

    main(args.obj_exp)
