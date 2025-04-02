from VGGNet import vggnet
import torch
from torchvision import datasets, transforms
from PGD import PGD_attack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])
subsampler = torch.utils.data.SubsetRandomSampler(range(100))
x = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transform),
    batch_size=1, sampler=subsampler)


model = vggnet(pretrained=True).to(device)
model.eval()

target = None

pgd_attack = PGD_attack(model, x , eps=0.3, eps_step=0.01, k=15, target=target)
pgd_attack.run()
pgd_attack.visualize()