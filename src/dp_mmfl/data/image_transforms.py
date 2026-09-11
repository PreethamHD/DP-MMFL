from torchvision import transforms

IMAGE_SIZE = 224
RESIZE_SIZE = 256

def get_train_transforms():
    """
    Image transformations used during model training.
    Preserves anatomical laterality by omitting horizontal flips as spatial positions are important for x-ryas.
    """
    return transforms.Compose([
        transforms.Resize(RESIZE_SIZE, antialias=True),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

def get_eval_transforms():
    """
    Deterministic transformations used for validation and evaluation.
    """
    return transforms.Compose([
        transforms.Resize(RESIZE_SIZE, antialias=True),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
