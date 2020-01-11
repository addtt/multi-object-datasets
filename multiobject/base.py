import numpy as np
from tqdm import tqdm


def generate_multiobject_dataset(n, shape, sprites, sprites_attr, count_distrib,
                                 allow_overlap=False):
    """
    Given a set of sprites, create a dataset with multiple sprites scattered
    on a black background of the specified size. It returns the images, an
    array with the number of objects per image, and a dictionary of
    (key=attribute_name, value=list of lists), containing attributes of the
    objects in each image.

    The distribution of the number of sprites is a parameter. This method
    generates images in order from the ones with more objects to the ones with
    fewer, and then shuffles the data set.

    Object locations are chosen at random until a valid location is picked.
    A valid location is such that the object is completely in the frame and,
    if allow_overlap=False, does not overlap with previously placed objects.

    :param n: required dataset size
    :param shape: tuple (H, W, channels), channels either 1 or 3.
    :param sprites: list of sprites (H, W, channels), channels either 1 or 3.
    :param sprites_attr: dict with key=attr_name and value=array_of_attributes
            for each sprite
    :param count_distrib: dict
    :param allow_overlap: bool
    :return: returns the dataset, an array with the number of objects per image,
            and a dictionary of (key=attribute_name, value=list of lists). For
            each attribute, the list has one list per image, and the inner lists
            have one attribute value per object in that image.
    """

    assert len(shape) == 3, "the image shape should be (height, width, channels)"
    probsum = sum(count_distrib.values())
    assert abs(probsum - 1.0) < 1e-6, "count probabilities must sum to 1"
    bgr = np.zeros(shape, dtype='int')
    color_channels = shape[-1]
    n_sprites = len(sprites)
    print("num sprites: {}".format(n_sprites))

    # Names of object attributes
    attribute_names = list(sprites_attr.keys())

    # Generated images
    images = []

    # Number of objects for each image
    n_objects = []

    # Dict: attribute name -> list of lists
    labels = {k: [] for k in attribute_names}

    # Dictionary with (key = n. of objects, value = n. of required images
    # with 'key' objects).
    # The sum of this might be larger than the required number of images,
    # but we just stop generating them when we have enough.
    counts = {k: int(np.ceil(v * n)) for k, v in count_distrib.items()}
    print("counts:", counts)

    # Sort count keys (= num objects)
    sorted_count_keys = sorted(counts.keys())

    for sprite in sprites:
        # Check sprite shape
        msg = ("each sprite should have shape (height, width, channels), "
               "found sprite with shape {}".format(sprite.shape))
        assert sprite.ndim == 3, msg
        msg = "sprites channels ({}) should be the same as background " \
              "channels ({}))".format(sprite.shape[-1], color_channels)
        assert color_channels == sprite.shape[-1], msg

    # Start from last key in the list: largest number of objects per image
    count_key_idx = len(sorted_count_keys) - 1

    generated_imgs = 0
    progress_bar = tqdm(total=n)
    while True:

        # Reached required number of images
        if generated_imgs >= n:
            break

        # If done generating images with this number of objects, go to the
        # next smallest required number of objects
        req_n_obj = sorted_count_keys[count_key_idx]
        if counts[req_n_obj] == 0:
            count_key_idx -= 1
            req_n_obj = sorted_count_keys[count_key_idx]

        # Start from background
        x = bgr.copy()

        # Locations containing rendered sprites
        occupied = np.zeros_like(x, dtype='uint8')

        # Pick the sprite type for each objects in this image
        random_obj_types = np.random.choice(
            range(n_sprites), size=req_n_obj, replace=True)

        # Number of placed objects
        obj_count = 0

        # Dictionary with (key=attribute name, value=list of attribute values
        # for each object in this image)
        image_labels = {k: [] for k in attribute_names}

        curr_attempts = 0   # current attempts to place objects
        while True:
            # Reached required number of objects
            if obj_count == req_n_obj:
                break

            # Hard limit, just drop sample at this point and try again
            if curr_attempts > 100:
                print("WARNING: too many attempts")
                break

            obj_type = random_obj_types[obj_count]
            obj_size = sprites[obj_type].shape
            r = np.random.randint(x.shape[0] - obj_size[0] + 1)
            c = np.random.randint(x.shape[1] - obj_size[1] + 1)
            curr_attempts += 1
            overlap = np.count_nonzero(
                occupied[r:r + obj_size[0], c:c + obj_size[1]]) > 0
            if overlap and not allow_overlap:
                continue
            occupied[r:r + obj_size[0], c:c + obj_size[1]] = 1

            # Render entity by adding and clipping
            sprite = sprites[obj_type]
            x[r:r + obj_size[0], c:c + obj_size[1]] += sprite
            x = np.clip(x, a_min=0, a_max=255)

            # Increment object counter
            obj_count += 1

            # For each sprite attribute, append the new object's attribute to
            # the attribute-specific list of attributes of previous objects in
            # the current image.
            for k in attribute_names:
                image_labels[k].append(sprites_attr[k][obj_type])

        # Hard limit, just drop sample at this point and try again
        if curr_attempts > 100:
            continue

        # Append image, number of objects in it, and each object's attributes
        images.append(x.astype('uint8'))
        n_objects.append(obj_count)
        counts[req_n_obj] -= 1
        for k in attribute_names:
            labels[k].append(np.array(image_labels[k]))

        generated_imgs += 1
        progress_bar.update()
    progress_bar.close()

    images = np.stack(images, axis=0)
    n_objects = np.array(n_objects)

    perm = np.random.permutation(len(images))  # indices
    images = images[perm]
    n_objects = n_objects[perm]
    for k in attribute_names:
        labels[k] = [labels[k][i] for i in perm]

    return images, n_objects, labels
