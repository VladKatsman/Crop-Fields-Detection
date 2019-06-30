import os
import json

import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import torch.utils.data
import torch.utils.data.distributed

import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
from tqdm import tqdm
from field_boundaries.utils.obj_factory import obj_factory
from field_boundaries.utils import seg_utils
from field_boundaries.utils.postprocessing import process_mask, BoundaryHandler
from google_maps_python.map_utils import polygon_to_url


# remove
Image.MAX_IMAGE_PIXELS = None


def main(exp_dir='/data/experiments', output_dir='/data/experiments', gpus=None, arch='resnet18', grid='/data',
         center=[0, 0]):

    # Check dir
    if not os.path.isdir(exp_dir):
        raise RuntimeError('Experiment directory was not found: \'' + exp_dir + '\'')
    warnings.filterwarnings("ignore", category=UserWarning)

    # Check CUDA device availability
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        gpus = list(range(torch.cuda.device_count())) if not gpus else gpus
        print('=> using GPU devices: {}'.format(', '.join(map(str, gpus))))
    else:
        gpus = None
        print('=> using CPU device')
    device = torch.device('cuda:{}'.format(gpus[0])) if gpus else torch.device('cpu')

    tensor_transforms = [transforms.ToTensor(),
                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    tensor_transforms = transforms.Compose(tensor_transforms)

    # Create model
    model = obj_factory(arch)
    model.to(device)

    # Load weights
    checkpoint_dir = exp_dir
    # model_path = os.path.join(checkpoint_dir, 'model_best.pth')
    model_path = os.path.join(checkpoint_dir, 'model_latest.pth')
    if os.path.isfile(model_path):
        print("=> loading checkpoint from '{}'".format(checkpoint_dir))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])

    # Support multiple GPUs
    if gpus and len(gpus) > 1:
        model = nn.DataParallel(model, gpus)

    # if input shape are the same for the dataset then set to True, otherwise False
    cudnn.benchmark = True

    # open grid
    grid = np.array(Image.open(grid))

    # evaluate
    validate(model, device, output_dir, grid, tensor_transforms, center)


def validate(model, device, output_dir, grid, tensor_transforms, center):

    # init post processing class
    crop = 21
    crop_size = 384
    step = 342
    height = grid.shape[0]
    num_crops_in_row = (height - crop_size) // step
    height_f = num_crops_in_row * step + crop_size
    center = center.split(',')
    center = float(center[0]), float(center[1])
    postproc = BoundaryHandler(center=center, grid_size=height, crop=crop)

    # init empty image
    final_res = np.zeros((height_f, height_f, 3)).astype(np.uint8)
    final_mask = np.zeros((height_f, height_f)).astype(np.uint8)
    json_file = {'polygons': []}

    # switch to evaluate mode
    model.train(False)
    with torch.no_grad():
        for i in tqdm(range(num_crops_in_row)):
            for j in range(num_crops_in_row):
                image = grid[i * step:crop_size + i * step,
                             j * step:crop_size + j * step]
                image = Image.fromarray(image.astype(np.uint8))
                inputs = tensor_transforms(image)
                inputs = inputs.unsqueeze(0)
                inputs = inputs.to(device)

                # compute output of the model
                output = model(inputs)

                # update metrics
                pred = output.data.max(1)[1].cpu().numpy()

                # opening/closing + find contours
                mask = pred[0]
                mask = np.array(mask)
                mask = process_mask(mask)
                if len(mask) == 0:
                    image = np.array(image)
                    image = image[crop:-crop, crop:-crop, :]
                    final_res[step * i:step * (i + 1), step * j: step * (j + 1), :] = image.astype(np.uint8)
                    continue
                # making mask colorfull
                color = [0, 255, 255]

                # using blending to concatenate mask and image
                img_with_mask = seg_utils.alpha_blend(image, color, mask, val=False)

                # crop mask (there is a memory error)
                mask = mask[crop:-crop, crop:-crop]
                final_mask[step * i:step * (i + 1), step * j: step * (j + 1)] = mask.astype(np.uint8)

                # crop image on edges
                img_crop = img_with_mask[crop:-crop, crop:-crop, :]
                final_res[step*i:step*(i+1), step*j: step*(j+1), :] = img_crop.astype(np.uint8)


    # crop boundaries
    final_res = final_res[crop:-crop, crop:-crop, :]  # probably to make sure no small segments are included

    # save blended image
    final_res = Image.fromarray(final_res.astype(np.uint8))
    final_res.save(os.path.join(output_dir, "result.jpg"))

    # find contours
    final_mask = final_mask[crop:-crop, crop:-crop].astype(np.uint8)
    json_file['polygons'] = postproc.get_coordinates(final_mask)

    # tmp
    # polygon_to_url(json_file['polygons'][100])

    # save json
    annotation_path_v = "{}/{}_{}.json".format(args.output_dir, center[0], center[1])
    with open(annotation_path_v, 'w') as output_json_file:
        json.dump(json_file, output_json_file)


if __name__ == "__main__":

    # Parse program arguments
    import argparse

    parser = argparse.ArgumentParser('Inference of segmentation algorithms')
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    parser.add_argument('exp_dir',
                        help='path to experiment directory')
    parser.add_argument('--output_dir', default=None, type=str, metavar='DIR',
                        help='path to directory to save predicted masks on images')
    parser.add_argument('--gpus', default=None, nargs='+', type=int, metavar='N',
                        help='list of gpu ids to use (default: all)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',  # choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('--grid', default=None, type=str, metavar='DIR',
                        help='path to directory with a grid image')
    parser.add_argument('--center',
                        help='coordinates of center point of the grid image')
    args = parser.parse_args()
    main(args.exp_dir, output_dir=args.output_dir, gpus=args.gpus, arch=args.arch, grid=args.grid, center=args.center)
