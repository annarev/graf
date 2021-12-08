# Loads a pre-trained Resnet-18 model.
import torch
import torchvision

class SketchFeatureExtractor:
    def __init__(self, checkpoint_path, device, feature_size=256):
        """Loads a pre-trained Resnet-18 model from checkpoint_path.

        The model must be trained using the @sthalles SimCLR implementation at:
        https://github.com/sthalles/SimCLR
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['state_dict']
        self.model = torchvision.models.resnet18(
                pretrained=False, num_classes=feature_size).to(device)

        # Layers before the last layer have 'backbone' prefix. We remove
        # this prefix here.
        compatible_state_dict = {}
        backbone_prefix = 'backbone.'
        backbone_prefix_len = len('backbone.')
        for key, value in state_dict.items(): 
          if key.startswith(backbone_prefix):
            new_key = key[backbone_prefix_len:]
            compatible_state_dict[new_key] = value

        status = self.model.load_state_dict(compatible_state_dict, strict=False)
        # Make sure the checkpoint loaded successfully
        assert status.missing_keys == ['fc.weight', 'fc.bias']
        assert status.unexpected_keys == ['fc.0.weight', 'fc.0.bias', 'fc.2.weight', 'fc.2.bias']

        # Remove the last layer.
        self.model.fc = torch.nn.Sequential()

        print('Loaded SimCLR pre-trained model')
        print(self.model.fc)

        self.model.eval()

    def get_features(self, sketch):
        features = self.model.forward(sketch)
        return features
