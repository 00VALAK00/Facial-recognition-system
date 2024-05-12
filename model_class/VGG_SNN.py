import torch
import torchvision
from torchvision.models import VGG19_Weights


# implementing the SNN with VGG encoder
class SNN_with_vgg(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        vgg = torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT).features
        self.vgg = torch.nn.Sequential(*list(vgg.children()))
        self.encoder = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size=(7, 7)),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=4096, out_features=768)
        )

    def forward(self, img1, img2):
        img1, img2 = img1.to(self.device).float(), img2.to(self.device).float(),
        x1 = self.encoder(self.vgg(img1))
        x2 = self.encoder(self.vgg(img2))
        d = torch.nn.PairwiseDistance()(x1,x2)
        d = d.unsqueeze(1)
        return d, x1, x2

    def get_transform(self):
        return VGG19_Weights.DEFAULT.transforms()





#%%
