from mobilenetv2 import mobilenet_v2
import torch
from torchvision import datasets, transforms
from FGSM import FGSM_attack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mobilenet = mobilenet_v2()
model = mobilenet_v2(pretrained = True)
model.to(device)

model.eval()

t = transforms.Compose([
            transforms.ToTensor(),
            ])
subsampler = torch.utils.data.SubsetRandomSampler(range(1000))
x = torch.utils.data.DataLoader(
    datasets.CIFAR10('cifar10/data', train=False, download=True, transform=t),
    batch_size=1, sampler=subsampler)

target = None

fgsm_attack = FGSM_attack(model, x, target, eps=0.15)

fgsm_attack.run()
fgsm_attack.visualize()