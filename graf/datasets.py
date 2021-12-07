import glob
import numpy as np
import os
from PIL import Image

from torchvision.datasets.vision import VisionDataset
from torchvision import transforms


class ImageDataset(VisionDataset):
    """
    Load images from multiple data directories.
    Folder structure: data_dir/filename.png
    """

    def __init__(self, data_dirs, transforms=None):
        # Use multiple root folders
        if not isinstance(data_dirs, list):
            data_dirs = [data_dirs]

        # initialize base class
        VisionDataset.__init__(self, root=data_dirs, transform=transforms)

        self.filenames = []
        root = []

        for ddir in self.root:
            filenames = self._get_files(ddir)
            self.filenames.extend(filenames)
            root.append(ddir)

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def _get_files(root_dir):
        return glob.glob(f'{root_dir}/*.png') + glob.glob(f'{root_dir}/*.jpg')

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img = Image.open(filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img


class Carla(ImageDataset):
    def __init__(self, *args, **kwargs):
        super(Carla, self).__init__(*args, **kwargs)


class CelebA(ImageDataset):
    def __init__(self, *args, **kwargs):
        super(CelebA, self).__init__(*args, **kwargs)


class CUB(ImageDataset):
    def __init__(self, *args, **kwargs):
        super(CUB, self).__init__(*args, **kwargs)
        

class Cats(ImageDataset):
    def __init__(self, *args, **kwargs):
      super(Cats, self).__init__(*args, **kwargs)
    
    @staticmethod
    def _get_files(root_dir):
      return glob.glob(f'{root_dir}/CAT_*/*.jpg')


class CelebAHQ(ImageDataset):
    def __init__(self, *args, **kwargs):
        super(CelebAHQ, self).__init__(*args, **kwargs)
    
    def _get_files(self, root):
        return glob.glob(f'{root}/*.npy')
    
    def __getitem__(self, idx):
        img = np.load(self.filenames[idx]).squeeze(0).transpose(1,2,0)
        if img.dtype == np.uint8:
            pass
        elif img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)
        else:
            raise NotImplementedError
        img = Image.fromarray(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return img

class ShapeNetSketchDataset(VisionDataset):
    """
    Load car view images and corresponding sketches.
    Folder structure:
       data_dir/input/sketch/id.png
       data_dir/other/id_angle.png
    """

    def __init__(self, data_dirs, transforms=None):
        assert (not isinstance(data_dirs, list),
                'ShapeNetSketchDataset only supports one directory')
        root_dir = os.path.dirname(data_dir)

        # initialize base class
        VisionDataset.__init__(self, root=root_dir, transform=transforms)

        self.filenames = sorted(self._get_car_view_files(data_dir))
        self.sketch_filenames = []

        sketch_data_dir = os.path.join(data_dir, 'input/sketch')

        # Add sketch filenames in the same order as image filenames.
        for view_filename in filenames:
          basename = os.path.basename(view_filename)
          # Remove the '_angle.png' suffix.
          basename = basename[:basename.find('_')]
          sketch_filename = os.path.join(sketch_data_dir, '%s.png' % basename)
          self.sketch_filenames.append(sketch_filename)
        print('Loaded %i examples.' % len(self.filenames))

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def _get_car_view_files(root_dir):
        return glob.glob(f'{root_dir}/other/*.png')

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        sketch_filename = self.sketch_filenames[idx]
        img = Image.open(filename).convert('RGB')
        sketch = Image.open(sketch_filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            sketch = self.transform(sketch)
        small_sketch = transforms.Resize(32)(sketch)
        return img, sketch, small_sketch

