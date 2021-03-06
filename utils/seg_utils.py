import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image


# Model layer wrappers
# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()


# METRICS

class RunningScore(object):
    """Used to compute our main metrics
        - overall accuracy
        - mean accuracy
        - mean IU
        - fwavacc

        It's better to be used at validation time only,
        because code is not optimized for Tensors and uses numpy
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.smooth = 1.0

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask] + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = (np.diag(hist).sum() + self.smooth) / (hist.sum() + self.smooth)
        acc_cls = (np.diag(hist) + self.smooth) / (hist.sum(axis=1) + self.smooth)
        acc_cls = np.nanmean(acc_cls)
        iu = (np.diag(hist) + self.smooth) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + self.smooth)
        mean_iu = np.nanmean(iu)
        freq = (hist.sum(axis=1) + self.smooth) / (hist.sum() + self.smooth)
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class Metrics(object):
    """  Used to compute our main metrics
        - overall accuracy
        - mean accuracy
        - mean IU
        - fwavacc
    Wrote in pytorch
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = torch.zeros((n_classes, n_classes))
        self.smooth = 1.0

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = torch.bincount(n_class * label_true[mask] + label_pred[mask],
                              minlength=n_class ** 2,
                              ).view(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix = self._fast_hist(
                lt.view(-1), lp.view(-1), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix.float()
        acc = (torch.diag(hist).sum() + self.smooth) / (hist.sum() + self.smooth)
        acc_cls = (torch.diag(hist) + self.smooth) / (hist.sum(dim=1) + self.smooth)
        acc_cls = torch.mean(acc_cls)
        iu = (torch.diag(hist) + self.smooth) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist) + self.smooth)
        mean_iu = torch.mean(iu)
        freq = (hist.sum(dim=1) + self.smooth) / (hist.sum() + self.smooth)
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = torch.zeros((self.n_classes, self.n_classes))


# INFERENCE

def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors

        Can be customized by the user,
        By default the mask is blue for idx 1 and black otherwise

    Returns:
        np.ndarray with dimensions (2, 3)
    """
    return np.asarray(
        [
            [0, 0, 0],
            [0, 100, 100],
        ]
    )


def decode_segmap(label_mask, n_classes,  label_colours=get_pascal_labels()):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        n_classes (int): number of classes in the dataset
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(1, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    return rgb


def alpha_blend(input_image, color, segmentation_mask, alpha=0.5, val=True):
    """Alpha Blending utility to overlay RGB masks on RBG images for open CV

    :param input_image: OPENCV
    :param segmentation_mask:
    :param alpha: is a  value
    :param

    Args:
        input_image: PIL.Image or np.ndarray
        segmentation_mask: PIL.Image or np.ndarray
        alpha (float): more val -> more weight to input image,
                       less val -> more weight to segmentation mask color
        weights: PIL.Image or np.ndarray, add smoothness to mask colors

    Returns:

    """
    if type(input_image) != np.ndarray:
        input_image = np.array(input_image).astype(np.uint8)
    if type(segmentation_mask) != np.ndarray:
        segmentation_mask = np.array(segmentation_mask).astype(np.uint8)
    alpha_mask = 1 - (segmentation_mask == 1) * alpha
    alpha_mask = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)
    if val:
        input_image = np.transpose(input_image, [1, 2, 0])
    blended = input_image * alpha_mask + color * (1-alpha_mask)

    return blended


def blend_seg_pred(img, seg, alpha=0.5):
    pred = seg.argmax(1)
    pred = pred.view(pred.shape[0], 1, pred.shape[1], pred.shape[2]).repeat(1, 3, 1, 1)
    blend = img

    for i in range(1, seg.shape[1]):
        color_mask = -torch.ones_like(img)
        color_mask[:, -i, :, :] = 1
        alpha_mask = 1 - (pred == i).float() * alpha
        blend = blend * alpha_mask + color_mask * (1 - alpha_mask)

    return blend


def blend_seg_label(img, seg, alpha=0.5):
    pred = seg.unsqueeze(1).repeat(1, 3, 1, 1)
    blend = img
    for i in range(1, pred.shape[1]):
        color_mask = -torch.ones_like(blend)
        color_mask[:, -i, :, :] = 1  # test colors
        alpha_mask = 1 - (pred == i).float() * alpha
        blend = blend * alpha_mask + color_mask * (1 - alpha_mask)

    return blend


def make_grid(input, pred, gt, seg, cols=8):
    imgs = []
    for d in range(0, input.size(0), cols):
        for i in range(d, min(d + cols, input.size(0))):
            imgs.append(input[i])
        for i in range(d, min(d + cols, input.size(0))):
            imgs.append(pred[i])
        for i in range(d, min(d + cols, input.size(0))):
            imgs.append(gt[i])
        # for i in range(d, min(d + cols, input.size(0))):
        #     imgs.append(seg[i])

    grid = vutils.make_grid(imgs, nrow=cols, normalize=True, scale_each=False)

    return grid


def pil_grid(images, max_horiz=np.iinfo(int).max):
    """ Concatenates images horizontally

    :param images: list of images to concatenate
    :param max_horiz: default is horizontal, 1 for vertical
    """
    n_images = len(images)
    n_horiz = min(n_images, max_horiz)
    h_sizes, v_sizes = [0] * n_horiz, [0] * (n_images // n_horiz)
    for i, im in enumerate(images):
        h, v = i % n_horiz, i // n_horiz
        h_sizes[h] = max(h_sizes[h], im.size[0])
        v_sizes[v] = max(v_sizes[v], im.size[1])
    h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
    im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
    for i, im in enumerate(images):
        im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
    return im_grid


if __name__ == "__main__":
    # Parse program arguments
    import argparse

    parser = argparse.ArgumentParser('get_bboxes_check')
    parser.add_argument('root', type=str, metavar='DIR',
                        help='paths to root directory')
    parser.add_argument('model', type=str, metavar='MODEL',
                        help='paths to trained model directory with .pt file')
    args = parser.parse_args()
