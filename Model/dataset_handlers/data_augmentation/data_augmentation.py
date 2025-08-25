# IMPORT

import torchvision
from torchvision import transforms

from models.Feature_extractor.LandmarkDetector.MTCNN import MTCNN_creation

########################################################################################################################

def my_transformations():
    my_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)
    ])

    return my_transforms

def default_resize_transformation():
    my_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return my_transform

def landmark_stream_transformations():
    my_transforms = transforms.Compose([
        transforms.ToPILImage(),
        landmarks_focus(),
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])

    return my_transforms

def features_stream_transformations():
    my_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)
    ])

    return my_transforms

########################################################################################################################
# MY TRANSFORMATION

class landmarks_focus(object):
    def __init__(self):
        pass

    def __call__(self, img):
        landmarks_extractor = MTCNN_creation()
        _, _, landmarks = landmarks_extractor.detect(img, landmarks=True)
        focused_image = landmarks_transformation(img, landmarks)

        return focused_image

    def __repr__(self):
        return self.__class__.__name__ + '()'

def landmarks_transformation(x, landmarks):

    # Extreme points of bounding box
    x_min = min(landmarks[0][i][0] for i in range(5))
    x_max = max(landmarks[0][i][0] for i in range(5))
    y_min = min(landmarks[0][i][1] for i in range(5))
    y_max = max(landmarks[0][i][1] for i in range(5))

    offset_left = 40
    offset_right = 40
    offset_top = 30
    offset_bottom = 10

    # Modifies extreme points
    x_min -= offset_left
    x_max += offset_right
    y_min -= offset_top
    y_max += offset_bottom

    # Cropping
    cropped_image = x.crop((x_min, y_min, x_max, y_max))

    return cropped_image
