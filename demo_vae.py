import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image

from multiobject import MultiObjectDataLoader
from multiobject.datasets import MultiDSpritesBinaryColor

epochs = 100
batch_size = 64
lr = 1e-3
dataset_filename = 'multi_dsprites_color_012.npz'


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        zdim = 32

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2, stride=4),
            nn.ELU(),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 64, 5, padding=2, stride=4),
            nn.ELU(),
            nn.Conv2d(64, 2 * zdim, 5, padding=2, stride=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(zdim, 64, 5, padding=2, stride=2, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(64, 64, 5, padding=1, stride=4, output_padding=1),
            nn.ELU(),
            nn.Dropout2d(0.25),
            nn.ConvTranspose2d(64, 3, 5, padding=0, stride=4, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        h, w = tuple(x.shape[2:])
        x = self.encoder(x)
        mu, lv = torch.chunk(x, 2, dim=1)
        std = (lv / 2).exp()
        z = torch.randn_like(mu) * std + mu
        out = self.decoder(z)
        out = out[:, :, :h, :w]  # crop 65 to 64
        return out, mu, lv


def main():

    path = os.path.join('generated', 'dsprites', dataset_filename)
    os.makedirs('demo_output', exist_ok=True)

    # Datasets and dataloaders
    print("loading dataset...")
    train_set = MultiDSpritesBinaryColor(path, train=True)
    test_set = MultiDSpritesBinaryColor(path, train=False)
    train_loader = MultiObjectDataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = MultiObjectDataLoader(test_set, batch_size=100)

    # Model and optimizer
    print("initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE().to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    # Training loop
    print("training starts")
    step = 0
    model.train()
    for e in range(1, epochs + 1):
        for x, labels in train_loader:

            # Run model and compute loss
            _, loss, recons, kl = forward(model, x, device)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % 100 == 0:
                print("[{}]  elbo: {:.2g}  recons: {:.2g}   kl: {:.2g}".format(
                    step, -loss.item(), recons.item(), kl.item()))

        # Test
        with torch.no_grad():
            model.eval()
            loss = recons = kl = 0.
            for x, labels in test_loader:
                out, loss_, recons_, kl_ = forward(model, x, device)
                k = len(x) / len(test_set)
                loss += loss_.item() * k
                recons += recons_.item() * k
                kl += kl_.item() * k
            model.train()
        print("TEST [epoch {}]  elbo: {:.2g}  recons: {:.2g}   kl: {:.2g}".format(
            e, -loss, recons, kl))

        n = 6
        nimg = n ** 2 // 2
        fname = os.path.join('demo_output', '{}.png'.format(e))
        imgs = torch.stack([x[:nimg], out[:nimg]])
        imgs = imgs.permute(1, 0, 2, 3, 4)
        imgs = imgs.reshape(n ** 2, x.size(1), x.size(2), x.size(3))
        save_image(imgs.cpu(), fname, nrow=n)


def forward(model, x, device):

    # Forward
    x = x.to(device)
    out, mu, lv = model(x)

    # Loss = -ELBO
    recons = F.binary_cross_entropy(out, x, reduction='none')
    kl = -0.5 * (1 + lv - mu ** 2 - lv.exp())
    recons = recons.sum((1, 2, 3)).mean()
    kl = kl.sum((1, 2, 3)).mean()
    loss = recons + kl

    return out, loss, recons, kl


if __name__ == '__main__':
    main()
