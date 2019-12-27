import numpy as np
from skimage.draw import polygon, ellipse


def get_ellipse(angle, color, scale, patch_size):
    img = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    rr, cc = ellipse(patch_size/2,
                     patch_size/2,
                     r_radius=scale*patch_size/2,
                     c_radius=scale*patch_size/3,
                     shape=img.shape,
                     rotation=angle)
    img[rr, cc, :] = color[None, None, :]
    return img

def get_square(angle, color, scale, patch_size):
    num_vert = 4
    return get_regular_polygon(angle, num_vert, color, scale, patch_size)

def get_triangle(angle, color, scale, patch_size):
    num_vert = 3
    return get_regular_polygon(angle, num_vert, color, scale, patch_size)

def get_regular_polygon(angle, num_vert, color, scale, patch_size):

    # Coordinates of starting vertex
    def x1(a): return (1 + np.cos(a) * scale) * patch_size / 2
    def y1(a): return (1 + np.sin(a) * scale) * patch_size / 2

    # Loop over circle and add vertices
    angles = np.arange(angle, angle + 2 * np.pi - 1e-3, 2 * np.pi / num_vert)
    coords = list(([x1(a), y1(a)] for a in angles))

    # Create image and set polygon to given color
    img = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    coords = np.array(coords)
    rr, cc = polygon(coords[:, 0], coords[:, 1], img.shape)
    img[rr, cc, :] = color[None, None, :]

    return img
