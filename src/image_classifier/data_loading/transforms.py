import torchvision.transforms as transforms  # type: ignore

class ImageProcessor:
    """Encapsulates image preprocessing and augmentation logic."""
    def __init__(self, image_size: int = 224, mean=None, std=None):
        self.image_size = image_size
        # Default to ImageNet standard
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

    def get_training_transforms(self) -> transforms.Compose:
        """Standard data augmentation and normalization pipeline for training."""
        return transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def get_validation_transforms(self) -> transforms.Compose:
        """Standard preprocessing pipeline for validation and inference."""
        resize_size = int((256 / 224) * self.image_size)
        return transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def get_inverse_transforms(self) -> transforms.Compose:
        """Pipeline to un-normalize a tensor back into a displayable PIL image."""
        inv_mean = [-m/s for m, s in zip(self.mean, self.std)]
        inv_std = [1/s for s in self.std]
        return transforms.Compose([
            transforms.Normalize(mean=inv_mean, std=inv_std),
            transforms.ToPILImage()
        ])
