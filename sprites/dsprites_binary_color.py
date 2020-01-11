import numpy as np

from utils import graphics


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
