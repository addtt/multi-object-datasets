import datetime

import matplotlib.pyplot as plt


def get_date_str():
    return datetime.datetime.now().strftime("%y%m%d_%H%M%S")

def show_img_grid(n, imgs, labels=None):
    plt.figure(figsize=(n * 2, n * 2))
    for i in range(n ** 2):
        plt.subplot(n, n, i + 1)
        plt.imshow(imgs[i].squeeze(), vmin=0, vmax=255)
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
        if labels is not None:
            lst = ["{} {}".format(k, labels[k][i]) for k in labels.keys()]
            print(list(lst))
    plt.show()
