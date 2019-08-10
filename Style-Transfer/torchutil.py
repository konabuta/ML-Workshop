import torch
from torchvision import transforms
import utils

def convert(test_img):
    with torch.no_grad():
        content_image = preprocess(test_img).numpy()
    return content_image

def preprocess(image_file):
    device = 'cpu'
    content_image = utils.load_image(image_file, size=625)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    return content_image
