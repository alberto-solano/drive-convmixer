import torch
import torch.nn as nn


class ActBnLayer(nn.Module):
    def __init__(self, h):
        super(ActBnLayer, self).__init__()
        self.bn_layer = nn.BatchNorm2d(h)

    def forward(self, x):
        return self.bn_layer(nn.functional.gelu(x))


class ConvMixerLayer(nn.Module):
    def __init__(self, h: int, kernel_size: int = 7):
        super(ConvMixerLayer, self).__init__()

        self.depthwise_conv = nn.Conv2d(h, h, kernel_size, groups=h, padding="same")
        self.actbn1 = ActBnLayer(h)
        self.pointwise_conv = nn.Conv2d(h, h, 1)
        self.actbn2 = ActBnLayer(h)

    def forward(self, x):
        x = self.actbn1(self.depthwise_conv(x)) + x
        x = self.pointwise_conv(x)
        return self.actbn2(x)


class ConvMixer(nn.Module):
    def __init__(self, h: int, depth: int = 10, patch_size: int = 4, kernel_size: int = 7, in_channels: int = 3):
        super(ConvMixer, self).__init__()

        self.embedding_layer = nn.Sequential(nn.Conv2d(in_channels, h, patch_size, stride=patch_size), ActBnLayer(h))
        self.convmixer_layers = nn.Sequential(
            *[ConvMixerLayer(h, kernel_size) for _ in range(depth)]
        )
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(h, 1, stride=patch_size, kernel_size=patch_size)#, output_padding=(0, 1))
        )

        print("The Net has {:,} parameters"
              .format(sum(w.numel() for w in self.state_dict().values())).replace(",", " "))

    def forward(self, x):
        x2 = self.embedding_layer(x)
        x2 = self.convmixer_layers(x2)
        x2 = self.output_layer(x2) + x.mean(axis=1, keepdim=True)
        return torch.sigmoid(x2)

    def check_dimensions(self, x):
        with torch.no_grad():
            x = self.embedding_layer(x)
            x = self.convmixer_layers(x)
            return {"embedding": x.shape, "output": self.output_layer(x).shape}