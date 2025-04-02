from VGGNet import vggnet
import torch
from torchvision import datasets, transforms
from FGSM import FGSM_attack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])
subsampler = torch.utils.data.SubsetRandomSampler(range(1000))
x = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transform),
    batch_size=1, sampler=subsampler)


model = vggnet(pretrained=True).to(device)
model.eval()

eps = 0.2
target = None

fgsm = FGSM_attack(model, x, target, eps)
fgsm.run()

fgsm.visualize()