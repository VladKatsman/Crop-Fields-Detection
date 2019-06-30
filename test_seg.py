import os
import random
import time

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
from field_boundaries.utils import utils
from field_boundaries.utils import seg_utils
from field_boundaries.utils.postprocessing import *


def main(exp_dir='/data/experiments', output_dir='/data/experiments', val_dir=None, workers=4,
         batch_size=1,  gpus=None, val_dataset=None,
         pil_transforms=None, tensor_transforms=None,
         arch='resnet18', cudnn_benchmark=True):

    # Validation
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

    # Initialize datasets
    if pil_transforms is not None:
        pil_transforms = [obj_factory(t) for t in pil_transforms]
        if len(pil_transforms) == 1:
            pil_transforms = pil_transforms[0]

    tensor_transforms = [obj_factory(t) for t in tensor_transforms] if tensor_transforms is not None else []
    if not tensor_transforms:
        tensor_transforms = [transforms.ToTensor(),
                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    tensor_transforms = transforms.Compose(tensor_transforms)

    val_dataset = obj_factory(val_dataset, val_dir, pil_transforms=pil_transforms,
                              tensor_transforms=tensor_transforms)

    # Initialize data loaders
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size,
        num_workers=workers)

    n_classes = val_dataset.n_classes

    # Create model
    model = obj_factory(arch)
    model.to(device)

    # Load weights
    checkpoint_dir = exp_dir
    model_path = os.path.join(checkpoint_dir, 'model_best.pth')
    if os.path.isfile(model_path):
        print("=> loading checkpoint from '{}'".format(checkpoint_dir))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])


    # Support multiple GPUs
    if gpus and len(gpus) > 1:
        model = nn.DataParallel(model, gpus)
    # if input shape are the same for the dataset then set to True, otherwise False
    cudnn.benchmark = cudnn_benchmark

    # metrics and criterion
    running_metrics = seg_utils.RunningScore(val_dataset.n_classes)
    criterion = nn.CrossEntropyLoss().to(device)

    # get names to save results
    file_list = tuple(open("{}/val_img.txt".format(val_dir), "r"))
    names = [id_.rstrip().split('/')[-1] for id_ in file_list]

    # evaluate on validation set
    validate(val_loader, model, device, output_dir, n_classes, batch_size, running_metrics, criterion, names)


def validate(val_loader, model, device, output_dir, n_classes, batch_size, running_metrics, criterion, names):
    batch_time = utils.AverageMeter()
    val_los = utils.AverageMeter()

    # init post processing class
    postproc = BoundaryHandler()

    # switch to evaluate mode
    model.train(False)
    with torch.no_grad():
        end = time.time()
        pbar = tqdm(val_loader, unit='batches')
        c = 0
        for j, (inputs, targets, images) in enumerate(pbar):

            val_inputs = inputs.to(device)
            val_targets = targets.to(device)

            # compute output and loss
            output = model(val_inputs)
            loss_sum = criterion(output, val_targets)

            # update metrics
            images = images.data.cpu().numpy()
            pred = output.data.max(1)[1].cpu().numpy()
            gt = val_targets.data.cpu().numpy()

            running_metrics.update(gt, pred)
            val_los.update(loss_sum.item())
            for i in range(pred.shape[0]):
                fname = names[c]
                c += 1
                # opening/closing + find contours
                mask = pred[i]
                mask = np.array(mask)
                mask = postproc.process_mask(mask)
                if len(mask) == 0:
                    continue
                # make mask be colorfull
                color = [0, 255, 255]

                # using blending to concatenate mask and image
                img_with_mask = seg_utils.alpha_blend(images[i], color, mask)
                img_with_mask = Image.fromarray(np.uint8(img_with_mask))

                # save blended image
                img_with_mask.save(os.path.join(output_dir, fname))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            pbar.set_description(
                'VALIDATION: '
                'Timing: [Batch: {batch_time.val:.3f} ({batch_time.avg:.3f})]; '.format(
                    len(val_loader)*batch_size, batch_time=batch_time))
        # Metrics
        score, class_iou = running_metrics.get_scores()

        # Epoch logs
        for k, v in score.items():
            print("{} {:.2f}".format(k, v))
        print("Average Validation loss: {:.3f}".format(val_los.avg))
        for k, v in class_iou.items():
            print("MeanIOU for class {:.2f} is {:.2f}".format(k, v))


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
    parser.add_argument('-t', '--train', type=str, metavar='DIR',
                        help='paths to train dataset root directory')
    parser.add_argument('-v', '--val', default=None, type=str, metavar='DIR',
                        help='paths to valuation dataset root directory')
    parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-i', '--iterations', default=None, type=int, nargs='+', metavar='N',
                        help='number of iterations per resolution to run')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--seed', default=None, type=int, metavar='N',
                        help='random seed')
    parser.add_argument('--gpus', default=None, nargs='+', type=int, metavar='N',
                        help='list of gpu ids to use (default: all)')
    parser.add_argument('-tb', '--tensorboard', action='store_true',
                        help='enable tensorboard logging')
    parser.add_argument('-vd', '--val_dataset', default=None, type=str, help='val dataset object')
    parser.add_argument('-pt', '--pil_transforms', default=None, type=str, nargs='+', help='PIL transforms')
    parser.add_argument('-tt', '--tensor_transforms', default=None, type=str, nargs='+', help='tensor transforms')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',  # choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('-cb', '--cudnn_benchmark', default=True, action='store_true',
                        help='if input shapes are the same for the dataset then set to True, otherwise False')
    parser.add_argument('-crf', '--crfpath', type=str, metavar='DIR',
                        help='paths to train dataset root directory')
    args = parser.parse_args()
    main(args.exp_dir, output_dir=args.output_dir, val_dir=args.val, workers=args.workers,
         batch_size=args.batch_size, gpus=args.gpus, val_dataset=args.val_dataset,
         pil_transforms=args.pil_transforms, tensor_transforms=args.tensor_transforms,
         arch=args.arch, cudnn_benchmark=args.cudnn_benchmark)
