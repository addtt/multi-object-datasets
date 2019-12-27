import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from multiobject import generate_multiobject_dataset
from multiobject.datasets import generate_dsprites
from utils import get_date_str

supported_sprites = ['dsprites']

def main():

    args = parse_args()

    ### SETTINGS #############################
    n = 100000   # num images
    frame_size = (64, 64)
    patch_size = 18

    # count_distrib = {1: 1}
    count_distrib = {0: 1/3, 1: 1/3, 2: 1/3}
    allow_overlap = True

    root = os.path.join('generated', args.dataset_type)
    ##########################################


    # Generate sprites and labels
    print("generating sprites...")
    if args.dataset_type == 'dsprites':
        sprites, labels = generate_dsprites(patch_size)
    else:
        raise NotImplementedError

    # Create dataset
    print("generating dataset...")
    img_shape = (*frame_size, 3)
    dataset, n_obj, labels = generate_multiobject_dataset(
        n, img_shape, sprites, labels,
        count_distrib=count_distrib,
        allow_overlap=allow_overlap)
    print("done")
    print("shape:", dataset.shape)

    # Number of objects is part of the labels
    labels['n_obj'] = n_obj

    # Save dataset
    print("saving...")
    os.makedirs(root, exist_ok=True)
    file_str = get_date_str()
    fname = 'multi_' + args.dataset_type + '_' + file_str
    fname = os.path.join(root, fname)
    np.savez_compressed(fname, x=dataset, labels=labels)
    print('done')

    # Show samples and print their attributes
    print("\nAttributes of shown samples:")
    plt.figure(figsize=(12, 12))
    n = 4
    for i in range(n ** 2):
        plt.subplot(n, n, i+1)
        plt.imshow(dataset[i])
        plt.gca().tick_params(
            axis='both',
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks are off on all edges
            right=False,
            bottom=False,
            top=False,
            labelleft=False,  # labels are off on all edges
            labelbottom=False)
        plt.subplots_adjust(top=.99, bottom=.01,
                            left=.01, right=.99,
                            wspace=.01, hspace=.01)
        print(list(["{} {}".format(k, labels[k][i]) for k in labels.keys()]))
    plt.show()

    # Show distribution of number of objects per image
    plt.figure()
    plt.hist(n_obj, np.arange(min(n_obj) - 0.5, max(n_obj) + 0.5 + 1, 1))
    plt.title("Distribution of num objects per image")
    plt.xlabel("Number of objects")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False)
    parser.add_argument('--type',
                        type=str,
                        default='dsprites',
                        metavar='NAME',
                        dest='dataset_type',
                        help="dataset type")
    args = parser.parse_args()
    if args.dataset_type not in supported_sprites:
        raise NotImplementedError(
            "unsupported dataset '{}'".format(args.dataset_type))
    return args


if __name__ == '__main__':
    main()
