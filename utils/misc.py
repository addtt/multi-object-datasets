import datetime

import matplotlib.pyplot as plt
import numpy as np


def get_date_str():
    return datetime.datetime.now().strftime("%y%m%d_%H%M%S")

def show_img_grid(n, imgs, labels=None, random_selection=False, fname=None):
    if random_selection:
        indices = np.random.choice(range(len(imgs)), n ** 2)
    else:
        indices = range(n ** 2)
    plt.figure(figsize=(n, n))
    for i in range(len(indices)):
        j = indices[i]
        plt.subplot(n, n, i + 1)
        if imgs[j].shape[-1] == 1:
            cmap = 'gray'
        else:
            cmap = None
        plt.imshow(imgs[j].squeeze(), vmin=0, vmax=255, cmap=cmap)
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
                            wspace=.04, hspace=.04)
        if labels is not None:
            lst = ["{} {}".format(k, labels[k][j]) for k in labels.keys()]
            print(list(lst))
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()
