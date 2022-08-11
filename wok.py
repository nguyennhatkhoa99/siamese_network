import torch
import io
from torchvision import datasets, transforms
from PIL import Image
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
        
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        img0 = Image.open(self.imgs[index][0])
        img0 = img0.convert("L")
        transform = transforms.Compose([
        transforms.ToTensor()])
        img0 = transform(img0)
        
        # # make a new tuple that includes original and the path
        # tuple_with_path = (img0, path)
        return img0, path, original_tuple[1]