import torch
from torch import nn
from torch.nn import functional as F


def reparameterize(mu, logvar):
    eps = torch.randn_like(mu)
    return mu + torch.exp(0.5 * logvar) * eps


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ConditionEncoder(nn.Module):
    def __init__(self, base_filters):
        super().__init__()
        f = int(base_filters)
        self.e1 = DownBlock(3, f)
        self.e2 = DownBlock(f, f * 2)
        self.e3 = DownBlock(f * 2, f * 4)
        self.e4 = DownBlock(f * 4, f * 8)
        self.bottleneck = DownBlock(f * 8, f * 8)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        bot = self.bottleneck(e4)
        return bot, e1, e2, e3, e4


class PosteriorEncoder(nn.Module):
    def __init__(self, base_filters, latent_dim, target_channels=1):
        super().__init__()
        f = int(base_filters)
        target_channels = int(target_channels)
        self.net = nn.Sequential(
            DownBlock(3 + target_channels, f),
            DownBlock(f, f * 2),
            DownBlock(f * 2, f * 4),
            DownBlock(f * 4, f * 8),
            DownBlock(f * 8, f * 8),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mu = nn.Linear(f * 8, int(latent_dim))
        self.logvar = nn.Linear(f * 8, int(latent_dim))

    def forward(self, image_a, target_b):
        x = torch.cat([image_a, target_b], dim=1)
        x = self.net(x).flatten(1)
        return self.mu(x), self.logvar(x)


class Decoder(nn.Module):
    def __init__(self, image_size, base_filters, latent_dim, dropout=0.1, target_channels=1):
        super().__init__()
        f = int(base_filters)
        self.image_size = int(image_size)
        self.f = f
        self.bot_side = self.image_size // 32
        self.z_proj = nn.Linear(int(latent_dim), self.bot_side * self.bot_side * f * 4)
        self.u4 = UpBlock(f * 8 + f * 4, f * 8, dropout=dropout)
        self.u3 = UpBlock(f * 8 + f * 8, f * 4)
        self.u2 = UpBlock(f * 4 + f * 4, f * 2)
        self.u1 = UpBlock(f * 2 + f * 2, f)
        self.u0 = UpBlock(f + f, max(f // 2, 16))
        self.out = nn.Conv2d(max(f // 2, 16), int(target_channels), 3, padding=1)

    def forward(self, bot, e1, e2, e3, e4, z):
        z_map = self.z_proj(z).view(z.shape[0], self.f * 4, self.bot_side, self.bot_side)
        x = torch.cat([bot, z_map], dim=1)
        x = self.u4(x)
        x = torch.cat([x, e4], dim=1)
        x = self.u3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.u2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.u1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.u0(x)
        return self.out(x)


class CVAE(nn.Module):
    def __init__(self, image_size, base_filters, latent_dim, dropout=0.1, target_channels=1):
        super().__init__()
        self.cond_encoder = ConditionEncoder(base_filters)
        self.target_channels = int(target_channels)
        self.posterior_encoder = PosteriorEncoder(base_filters, latent_dim, target_channels=self.target_channels)
        self.decoder = Decoder(
            image_size,
            base_filters,
            latent_dim,
            dropout=dropout,
            target_channels=self.target_channels,
        )
        self.latent_dim = int(latent_dim)

    def forward_train(self, image_a, target_b, latent_mode="posterior"):
        bot, e1, e2, e3, e4 = self.cond_encoder(image_a)
        mu, logvar = self.posterior_encoder(image_a, target_b)
        latent_mode = str(latent_mode).lower()
        if latent_mode in {"zero", "zeros", "infer", "inference"}:
            z = torch.zeros_like(mu)
        elif latent_mode in {"random", "prior"}:
            z = torch.randn_like(mu)
        else:
            z = reparameterize(mu, logvar)
        logits = self.decoder(bot, e1, e2, e3, e4, z)
        return logits, mu, logvar

    def forward_infer(self, image_a, z=None):
        bot, e1, e2, e3, e4 = self.cond_encoder(image_a)
        if z is None:
            z = torch.zeros((image_a.shape[0], self.latent_dim), dtype=image_a.dtype, device=image_a.device)
        logits = self.decoder(bot, e1, e2, e3, e4, z)
        return logits


class CVAEInference(CVAE):
    def predict(self, image_a, z=None):
        return torch.sigmoid(self.forward_infer(image_a, z=z))
