import os

import numpy as np
from skimage.transform import resize


def generate_binary_mnist(patch_size,
                          binarize_mode='rounding',
                          anti_aliasing=True):
    """
    Reads original MNIST training set from a .npz file, rescales it with
    bilinear interpolation, and statically binarizes it by either rounding or
    sampling.

    :param patch_size: size of the output MNIST digits (original is 28)
    :param binarize_mode: (str) 'rounding' | 'sampling'. Default: rounding.
            We're usually downscaling so we get very low resolution images, and
            then samples would suck.
    :param anti_aliasing: (bool) default: True
    :return: 1) list of binarized rescaled MNIST digits, 2) attribute dictionary
            containing only a 'labels' item, which is a list of labels (classes)
    """

    # Read archive with original MNIST dataset
    mnist_file = os.path.join('_data', 'mnist', 'original_mnist.npz')
    npz = np.load(mnist_file)

    # Only keep training digits
    imgs = npz['x_train']
    labels = npz['labels_train']
    assert imgs.shape == (60000, 28, 28, 1)
    assert labels.shape == (60000,)

    # Normalize
    imgs = np.float32(imgs) / 255

    # Resize images
    if patch_size != 28:
        imgs = resize(
            imgs,
            (imgs.shape[0], patch_size, patch_size, 1),
            order=3,
            mode='constant',
            anti_aliasing=anti_aliasing)

    # Binarize by rounding or by sampling
    if binarize_mode == 'rounding':
        imgs = np.round(imgs)
    elif binarize_mode == 'sampling':
        r = np.random.random_sample(imgs.shape)
        imgs = np.int32(r < imgs)
    else:
        raise ValueError("unknown binarize mode '{}'".format(binarize_mode))

    # Back to [0, 255] uint8
    imgs = imgs * 255
    imgs = imgs.astype('uint8')

    # Sprite attributes
    attr = {
        'labels': labels
    }

    return imgs, attr
