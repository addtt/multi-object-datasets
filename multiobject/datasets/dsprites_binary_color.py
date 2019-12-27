import numpy as np
import torch
from torch.utils.data import Dataset

from utils import graphics


class MultiDSpritesBinaryColor(Dataset):

    def __init__(self, data_path, train, split=0.9):
        super().__init__()

        # Load data
        data = np.load(data_path, allow_pickle=True)

        # Rescale and transpose images
        x = np.array(data['x'], dtype=np.float32) / 255
        x = np.transpose(x, [0, 3, 1, 2])  # batch, channels, h, w
        assert x.shape[1] == 3

        # Get labels
        labels = data['labels'].item()

        # Split train and test
        split = int(split * len(x))
        if train:
            indices = range(split)
        else:
            indices = range(split, len(x))

        # From numpy/ndarray to torch tensors (labels are lists of tensors as
        # they might have different sizes)
        self.x = torch.from_numpy(x[indices])
        self.labels = self._labels_to_tensorlist(labels, indices)

    @staticmethod
    def _labels_to_tensorlist(labels, indices):
        out = {k: [] for k in labels.keys()}
        for i in indices:
            for k in labels.keys():
                t = labels[k][i]
                t = torch.as_tensor(t)
                out[k].append(t)
        return out

    def __getitem__(self, index):
        x = self.x[index]
        labels = {k: self.labels[k][index] for k in self.labels.keys()}
        return x, labels

    def __len__(self):
        return self.x.size(0)




def generate_dsprites(patch_size, num_colors=7, num_angles=40, num_scales=6):
    assert num_colors in [1, 7]
    assert num_angles > 0
    assert num_scales > 0
    min_scale = .5
    max_scale = 1.

    angles = np.linspace(0, 2 * np.pi * (1 - 1 / num_angles), num_angles)
    scales = np.linspace(min_scale, max_scale, num_scales)
    colors = np.array([
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
    ])[:num_colors]

    # List of len (num_colors * num_angles * num_scales * 3)
    # Each element has shape (patch size, patch size, 3)
    sprite_imgs = []
    sprite_shapes = []
    sprite_angles = []
    sprite_colors = []
    sprite_scales = []
    for scale in scales:
        for color in colors:
            for angle in angles:
                img = graphics.get_square(angle, color, scale, patch_size)
                sprite_imgs.append(img)
                img = graphics.get_triangle(angle, color, scale, patch_size)
                sprite_imgs.append(img)
                img = graphics.get_ellipse(angle, color, scale, patch_size)
                sprite_imgs.append(img)
                sprite_shapes.extend([0, 1, 2])
                sprite_angles.extend([angle]*3)
                sprite_colors.extend([color]*3)
                sprite_scales.extend([scale]*3)

    attr = {
        'shape': sprite_shapes,
        'angle': sprite_angles,
        'color': sprite_colors,
        'scale': sprite_scales,
    }

    return sprite_imgs, attr
