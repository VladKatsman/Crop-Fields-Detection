import os
import json
import time

import warnings
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import torch.utils.data
import torch.utils.data.distributed

import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
from multiprocessing import Pool
from utils.obj_factory import obj_factory
from utils import seg_utils
from utils.postprocessing import process_mask, BoundaryHandler
from utils.data_utils import create_grid_prod


# remove
Image.MAX_IMAGE_PIXELS = None


def main(model_path='/data/experiments', output_dir='/data/experiments', gpus=None, arch='resnet18', grid_name=1,
         points=[[10, 20, 0, 20], [40, 20, 40, 0]], size=384, stride=342, batch_size=16, num_workers=10):

    # Check dir
    if not os.path.isdir(model_path):
        raise RuntimeError('Experiment directory was not found: \'' + model_path + '\'')
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
    checkpoint_dir = model_path
    # model_path = os.path.join(checkpoint_dir, 'model_best.pth')  # predicts sand areas as fields
    model_path = os.path.join(checkpoint_dir, 'model_latest.pth')  # doens predict sand areas as fields
    if os.path.isfile(model_path):
        print("=> loading checkpoint from '{}'".format(checkpoint_dir))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])

    # Support multiple GPUs
    if gpus and len(gpus) > 1:
        model = nn.DataParallel(model, gpus)

    # eval mode
    model.eval()
    torch.set_grad_enabled(False)

    # if input shape are the same for the dataset then set to True, otherwise False
    cudnn.benchmark = True

    # convert points string to list of floats
    points = [float(p) for p in points.split(',')]

    # download grid
    # p1 = time.time()
    grids_container, center, h_grid_out, w_grid_out = create_grid_prod(points,
                                                                       output_path="{}/grids".format(output_dir),
                                                                       name=grid_name)

    # preprocess grid and initiate constants
    size = size
    stride = stride
    crop = (size - stride) // 2
    batch_size = batch_size

    # we can process data in batches of 16 more efficiently
    H = W = 5514  # shape of the image in order to get 16x16 crops of size=384
    H1 = W1 = 5514 - crop * 2

    # initiate CPU post-processing module
    post_proc = BoundaryHandler(center, simplify_thresh=0.01, zoom=18)
    pool = Pool(processes=num_workers)

    # initiate polygons container
    json_file = {'polygons': []}

    # iterate over grid
    N = len(grids_container)
    for n in range(N):

        # open grid and coordinates anchors to regain actual coordinate values from the pixel values
        grid_path, h_cor, w_cor = grids_container[n]
        grid = cv2.imread(grid_path)[:,:,::-1]  # swap BGR to RGB

        # check for padding (we pad bot and right) in order to split to batches of 16 images
        h, w = grid.shape[:2]
        h_pad = (H % h)
        w_pad = (W % w)

        # move grid to PIL in order to use further (requires some memory)
        grid = Image.fromarray(grid.astype(np.uint8))

        # move grid to tensor, normalization
        tensor = tensor_transforms(grid)

        # if pad is not zero
        if h_pad or w_pad:
            tensor = F.pad(tensor, (0, w_pad, 0, h_pad))

        # split tensor to patches and patches to batches
        patches = tensor.unfold(1, size, stride).unfold(2, size, stride)  # should work for the tensor C x H x W

        # permute in order to retrieve information further
        patches = patches.permute(1, 2, 0, 3, 4)

        # create empty tensor
        gpu_container = torch.empty(batch_size, batch_size, size, size)

        # GPU part
        # p1 = time.time()
        for i in range(batch_size):
            input = patches[i].to(device)
            output = model(input)
            preds = output.data.max(1)[1].cpu()
            # transfer preds from GPU to CPU and send predictions to container
            gpu_container[i] = preds
        # p2 = time.time() - p1

        # CPU part
        cpu_container = np.zeros((16, 16, 384, 384)).astype(np.uint8)
        predictions = np.array(gpu_container).astype(np.uint8)
        for i in range(batch_size):
            res = pool.map(process_mask, predictions[i])
            if len(cpu_container) == 0:
                # there is no farms found
                continue
            cpu_container[i] = np.stack(res, axis=0)

        # remove grids overlapping introduced before and reconstruct area of the sub grid
        result = torch.from_numpy(cpu_container)
        result = result[:, :, crop:-crop, crop:-crop]
        result = result.permute(0, 2, 1, 3).contiguous().view(H1, W1)

        # remove predictions from zero-pad area (bot and right)
        result = result[:H1 - h_pad, :W1 - w_pad]

        # remove predictions from the last padded sub grids if there are
        if h_grid_out and h != 5120:
            result = result[:, :h_grid_out, :]
        if w_grid_out and w != 5120:
            result = result[:, :, w_grid_out]

        # drop very small sub grids
        if result.shape[0] < 3 or result.shape[1] < 3:
            continue
        # finally find contours and polygons
        array = np.array(result).astype(np.uint8)
        polygons = post_proc.get_coordinates(array, h_cor, w_cor)
        json_file['polygons'].append(polygons)

        # tmp draw result for debug
        # color = [0, 255, 255]
        #
        # # using blending to concatenate mask and image
        # image = np.array(grid).astype(np.uint8)
        # image = image[crop:-crop, crop:-crop, :]
        # mask = np.array(result).astype(np.uint8)
        # img_with_mask = seg_utils.alpha_blend(image, color, mask, val=False)
        #
        # final_res = Image.fromarray(img_with_mask.astype(np.uint8))
        # save_name = grid_path.split('/')[-1]
        # final_res.save(os.path.join(output_dir, "result_{}".format(save_name)))

    # save json file
    annotation_path_v = "{}/{}_{}.json".format(output_dir, center[0], center[1])
    with open(annotation_path_v, 'w') as output_json_file:
        json.dump(json_file, output_json_file)
    # print(time.time() - p1)


if __name__ == "__main__":

    # Parse program arguments
    import argparse

    parser = argparse.ArgumentParser('Inference of segmentation algorithms')
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    parser.add_argument('--model_path',
                        help='path to directory with a trained model')
    parser.add_argument('--output_dir', default=None, type=str, metavar='DIR',
                        help='path to directory to save predicted masks on images')
    parser.add_argument('--gpus', default=None, nargs='+', type=int, metavar='N',
                        help='list of gpu ids to use (default: all)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',  # choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('--grid_name', default=None, type=str, metavar='DIR',
                        help='name of the grid file')
    parser.add_argument('--points',
                        help='list of lists of coordinates in the next format [[x0, y0, x1, y1],...]')
    args = parser.parse_args()
    main(model_path=args.model_path, output_dir=args.output_dir, gpus=args.gpus, arch=args.arch, grid_name=args.grid_name,
         points=args.points)
