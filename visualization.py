# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from matplotlib import pyplot as plt
import colorcet as cc
from PIL import Image
import torchvision
import torch
import torch.nn as nn
import cv2

def VisualizeImageGrayscale(image_3d, percentile=99):
  r"""Returns a 3D tensor as a grayscale 2D tensor.

  This method sums a 3D tensor across the absolute value of axis=2, and then
  clips values at a given percentile.
  """
  image_2d = np.sum(np.abs(image_3d), axis=2)

  vmax = np.percentile(image_2d, percentile)
  vmin = np.min(image_2d)

  return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)

def VisualizeImageDiverging(image_3d, percentile=99):
  r"""Returns a 3D tensor as a 2D tensor with positive and negative values.
  """
  image_2d = np.sum(image_3d, axis=2)

  span = abs(np.percentile(image_2d, percentile))
  vmin = -span
  vmax = span

  return np.clip((image_2d - vmin) / (vmax - vmin), -1, 1)

def normalize(mask, vmin=None, vmax=None, percentile=99):
    if vmax is None:
        vmax = np.percentile(mask, percentile)
    if vmin is None:
        vmin = np.min(mask)
    return (mask - vmin) / (vmax - vmin + 1e-10)


def make_grayscale(mask):
    return np.sum(mask, axis=2)


def make_black_white(mask):
    return make_grayscale(np.abs(mask))


def show_mask(mask, title='', cmap=None, alpha=None, norm=True, axis=None, vmin_max=None):
    if norm:
        mask = normalize(mask)
    if vmin_max is not None:
        (vmin, vmax) = vmin_max
    else:
        (vmin, vmax) = (-1, 1) if cmap == cc.cm.bkr else (0, 1)
    if axis is None:
        plt.imshow(mask, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax, interpolation='nearest')
        if title:
            plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        axis.imshow(mask, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax, interpolation='nearest')
        if title:
            axis.set_title(title)
        axis.axis('off')


def cut_image_with_mask(image_path, mask, title='', percentile=70, axis=None):
    image = np.moveaxis(load_image(image_path, size=mask.shape[0], preprocess=False).numpy().squeeze(), 0, -1)
    mask = mask > np.percentile(mask, percentile)
    image[~mask] = 0

    if axis is None:
        plt.imshow(image, interpolation='lanczos')
        if title:
            plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        axis.imshow(image, interpolation='lanczos')
        if title:
            axis.set_title(title)
        axis.axis('off')


def show_mask_on_image(image, mask, title='', cmap=cc.cm.bmy, alpha=0.7, axis=None):
    # image = load_image(image_path, size=mask.shape[0], color_mode='L', preprocess=False).numpy().squeeze()
    if axis is None:
        plt.imshow(image, cmap=cc.cm.gray, interpolation='lanczos')
    else:
        axis.imshow(image, cmap=cc.cm.gray, interpolation='lanczos')
    show_mask(mask, title, cmap, alpha, norm=False, axis=axis)


def pil_loader(path, color_mode='RGB'):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(color_mode)


def load_image(path, size=None, color_mode='RGB', preprocess=True):
    pil_image = pil_loader(path, color_mode)
    shape = np.array(pil_image).shape
    transform_list = list()
    if size is not None and size != shape[0]:
        if size < shape[0]:
            if size < 256 < shape[0]:
                transform_list.append(torchvision.transforms.Resize(256))
            transform_list.append(torchvision.transforms.CenterCrop(size))
        else:
            print(f"Warning: Desired size {size} larger than image size {shape[0]}x{shape[1]}. Upscaling.")
            transform_list.append(torchvision.transforms.Resize(size))
    transform_list.append(torchvision.transforms.ToTensor())
    if preprocess:
        transform_list.append(torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transform = torchvision.transforms.Compose(transform_list)
    return transform(pil_image).unsqueeze(0)

def anomaly_map_to_color_map(anomaly_map: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute anomaly color heatmap.

    Args:
        anomaly_map (np.ndarray): Final anomaly map computed by the distance metric.
        normalize (bool, optional): Bool to normalize the anomaly map prior to applying
            the color map. Defaults to True.

    Returns:
        np.ndarray: [description]
    """
    if normalize:
        anomaly_map = (anomaly_map - anomaly_map.min()) / np.ptp(anomaly_map)
    anomaly_map = anomaly_map * 255
    anomaly_map = anomaly_map.astype(np.uint8)

    anomaly_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    anomaly_map = cv2.cvtColor(anomaly_map, cv2.COLOR_BGR2RGB)
    return anomaly_map


def superimpose_anomaly_map(
    anomaly_map: np.ndarray, image: np.ndarray, alpha: float = 0.4, gamma: int = 0, normalize: bool = False
) -> np.ndarray:
    """Superimpose anomaly map on top of in the input image.

    Args:
        anomaly_map (np.ndarray): Anomaly map
        image (np.ndarray): Input image
        alpha (float, optional): Weight to overlay anomaly map
            on the input image. Defaults to 0.4.
        gamma (int, optional): Value to add to the blended image
            to smooth the processing. Defaults to 0. Overall,
            the formula to compute the blended image is
            I' = (alpha*I1 + (1-alpha)*I2) + gamma
        normalize: whether or not the anomaly maps should
            be normalized to image min-max


    Returns:
        np.ndarray: Image with anomaly map superimposed on top of it.
    """

    anomaly_map = anomaly_map_to_color_map(anomaly_map.squeeze(), normalize=normalize)
    superimposed_map = cv2.addWeighted(anomaly_map, alpha, image, (1 - alpha), gamma)
    return superimposed_map