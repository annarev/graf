import torch
import torch.nn as nn

def GetImageFeatureModel():
    """Creates a model that maps a 32x32 image to a 384 feature vector.
    """
    padding = 1
    blocks = [
        # 3 x 32 x 32 ==> 6 x 16 x 16
        nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4, stride=2, padding=padding),
        nn.BatchNorm2d(6),
        nn.LeakyReLU(0.2),
        # 6 x 16 x 16 ==> 12 x 8 x 8
        nn.Conv2d(in_channels=6, out_channels=12, kernel_size=4, stride=2, padding=padding),
        nn.BatchNorm2d(12),
        nn.LeakyReLU(0.2),
        # 12 x 8 x 8 ==> 24 x 4 x 4 
        nn.Conv2d(in_channels=12, out_channels=24, kernel_size=4, stride=2, padding=padding),
        nn.BatchNorm2d(24),
        nn.LeakyReLU(0.2),
        nn.Flatten()
    ]
    return nn.Sequential(*blocks)

def GetHeadModel():
    """Creates a model that maps sketch and generated image features to
    a single value for classification.
    """
    combined_blocks = [
            nn.Linear(24 * 4 * 4 + 512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
    ]
    return nn.Sequential(*combined_blocks)


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, imsize=64, hflip=False):
        super(Discriminator, self).__init__()
        self.nc = nc
        assert(imsize==32)  # Currently only support 32x32 images
        self.imsize = imsize
        self.image_feature_net = GetImageFeatureModel()
        self.head_net = GetHeadModel()

    def forward(self, input, y=None):
        # The input is of type discriminator_input.DiscriminatorInput.
        # Reshape input image: (BxN_samples)xC -> BxCxHxW
        img_nchw = input.img.view(-1, self.imsize, self.imsize, 3).permute(0, 3, 1, 2)
        img_features = self.image_feature_net(img_nchw)
        # Concatenate image and sketch features and pass them to the head model.
        combined_features = torch.cat([img_features, input.sketch_features], dim=-1)
        return self.head_net(combined_features)

    def train(self):
        self.image_feature_net.train()
        self.head_net.train()

    def eval(self):
        self.image_feature_net.eval()
        self.head_net.eval()

