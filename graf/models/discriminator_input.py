"""Defines an image, sketch_feature pair object to pass to discriminator."""

class DiscriminatorInput:

    def __init__(self, img, sketch_features):
        self.img = img
        self.sketch_features = sketch_features
        self.sketch_features.detach()

    def requires_grad_(self):
        self.img.requires_grad_()
        self.sketch_features.requires_grad_()

    def size(self, dim=None):
        if dim is not None:
          return self.img.size(dim)
        return self.img.size()

    def __iter__(self):
        yield self.img
        yield self.sketch_features
