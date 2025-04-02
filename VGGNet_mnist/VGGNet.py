import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VGG(nn.Module):
    def __init__(self, features, output_dim):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 256), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.reshape(x.shape[0], -1)
        x = self.classifier(h)
        return x


def get_vgg_layer(config, batch_norm):
    layers = []
    in_channels = 1

    for c in config:
        assert c =='M' or isinstance(c, int)
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size = 2)]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding= 1)
            if batch_norm:
                layers +=[conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace = True)]
            else:
                layers +=[conv2d, nn.ReLU(inplace = True)]

            in_channels = c
    return nn.Sequential(*layers)


def vggnet(pretrained=False, progress=True, device="cpu", config='vgg11', batch_norm=True, output_dim=10, **kwargs):
    vgg_configs = {
        'vgg11': [64, 'M',128, 'M', 256, 256, 'M',512,512, 'M',512,512, 'M']
    }

    features = get_vgg_layer(vgg_configs[config], batch_norm=batch_norm)
    model = VGG(features, output_dim=output_dim)

    if pretrained:

        state_dict = torch.load("pretrained_model/vgg_mnist.pt", map_location=device)
        model.load_state_dict(state_dict)

    return model