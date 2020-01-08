import argparse
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.adamax import Adamax

from multiobject import MultiObjectDataLoader, MultiObjectDataset

epochs = 100
batch_size = 64
lr = 3e-4
dataset_filename = os.path.join(
    'dsprites',
    'multi_dsprites_color_012.npz')
# dataset_filename = os.path.join(
#     'binary_mnist',
#     'multi_binary_mnist_012.npz')


class SimpleBlock(nn.Module):
    def __init__(self, ch, kernel, stride=1, dropout=0.25):
        super().__init__()
        assert kernel % 2 == 1
        padding = (kernel - 1) // 2
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, kernel, padding=padding, stride=stride),
            nn.Dropout2d(dropout),
            nn.LeakyReLU(),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    def __init__(self, color_channels, n_classes):
        super().__init__()

        self.convnet = nn.Sequential(
            nn.Conv2d(color_channels, 64, 5, padding=2, stride=2),
            nn.LeakyReLU(),
            SimpleBlock(64, 3, stride=2),
            SimpleBlock(64, 3, stride=2),
            SimpleBlock(64, 3, stride=2),
            nn.Conv2d(64, 64, 3, padding=1, stride=2),
        )
        self.fcnet = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.convnet(x)  # output is 2x2 for 64x64 images
        x = x.sum((2, 3))    # sum over spatial dimensions
        x = self.fcnet(x)
        return x


def main():

    args = parse_args()

    path = os.path.join('generated', args.dataset_path)

    # Datasets and dataloaders
    print("loading dataset...")
    train_set = MultiObjectDataset(path, train=True)
    test_set = MultiObjectDataset(path, train=False)
    train_loader = MultiObjectDataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = MultiObjectDataLoader(test_set, batch_size=100)

    # Model and optimizer
    print("initializing model...")
    channels = train_set.x.shape[1]
    n_classes = 3   # hardcoded for dataset with 0 to 2 objects
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(channels, n_classes).to(device)
    optimizer = Adamax(model.parameters(), lr=lr)

    # Training loop
    print("training starts")
    step = 0
    model.train()
    for e in range(1, epochs + 1):
        for x, labels in train_loader:

            # Run model and compute loss
            loss, acc = forward(model, x, labels, device)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % 100 == 0:
                print("[{}]  loss: {:.2g}  acc: {:.2g}".format(
                    step, loss.item(), acc))

        # Test
        with torch.no_grad():
            model.eval()
            loss = acc = 0.
            for x, labels in test_loader:
                loss_, acc_ = forward(model, x, labels, device)
                k = len(x) / len(test_set)
                loss += loss_.item() * k
                acc += acc_ * k
            model.train()
        print("TEST [epoch {}]  loss: {:.2g}  acc: {:.2g}".format(
            e, loss, acc))



def forward(model, x, labels, device):

    # Forward pass through model
    n = labels['n_obj'].to(device)
    x = x.to(device)
    logits = model(x)

    # Loss
    loss = F.cross_entropy(logits, n)

    # Accuracy
    pred = logits.max(1)[1]
    accuracy = (n == pred).float().mean().item()

    return loss, accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False)
    parser.add_argument('--dataset',
                        type=str,
                        default=dataset_filename,
                        metavar='PATH',
                        dest='dataset_path',
                        help="relative path of the dataset")
    return parser.parse_args()


if __name__ == '__main__':
    main()
