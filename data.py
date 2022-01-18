import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.datasets as dset


def readCocoLabels(file_path: str):

    labels_dict, counter = {}, 1

    labels_file = open(file_path)
    for label in labels_file:
        labels_dict[counter] = label.strip()
        counter += 1

    return labels_dict


def categoryLabel(id, label_dict):
    return label_dict[id]


"""
Creating a class for Coco data
make sure to download the coco-2017 train and val datasets
- root directory should be -> "./data/COCO"
- annFile path should be -> "./data/COCO/annotations"
"""


class CocoData(Dataset):

    def __init__(self, rootDirectory: str,  annFile: str, labels_path, device, split='train', transform=None):

        super(CocoData, self).__init__()

        # Make sure that the value of split is either train or validation
        self.split = split.upper()
        assert self.split in {'TRAIN', 'VALIDATION'}

        self.device = device
        self.labels = readCocoLabels(labels_path)
        self.transform = transform
        self.coco_dataset = None

        if self.split == 'TRAIN':
            root = os.path.join(rootDirectory, 'train2017')
            annFile = os.path.join(annFile, 'instances_train2017.json')
        else:
            root = os.path.join(rootDirectory, 'val2017')
            annFile = os.path.join(annFile, 'instances_val2017.json')

        self.coco_dataset = dset.CocoDetection(root=root, annFile=annFile)

    def get_items(self, y, item):
        items = []
        for i in y:
            items.append(i[item])

        return torch.tensor(items)

    def __getitem__(self, index: int):

        image, targets = self.coco_dataset[index]
        size = transforms.ToTensor()(image).size()
        if self.transform is not None:
            image, targets = self.transform(image, targets, self.labels)
        if len(targets) == 0:
            return self.__getitem__(index + 1)

        bboxes = self.get_items(targets, 'bbox')
        labels = self.get_items(targets, 'category_id')

        return image, bboxes, labels

    def __len__(self) -> int:
        return len(self.coco_dataset)

    def collate_fn(self, batch):

        images = []
        bboxes = []
        labels = []

        for b in batch:
            images.append(b[0])
            bboxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, bboxes, labels


def CustomCocoTransform(image, target, labels):
    """
     - transfroms the PIL image into Pytorch tensor
     - retreive the width and hight of the image
     - resize the image to size of 300 x 300
     - generate a new annotation for the image
     - this new annotation keeps the id, the category_id, and the bbox
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    new_size = [300, 300]

    ToTensor = transforms.ToTensor()
    image = ToTensor(image)

    height, width = image.shape[1], image.shape[2]
    Resize = transforms.Resize(new_size)

    compose = transforms.Compose([transforms.Normalize(mean, std), Resize])

    image = compose(image)

    new_target = []

    for objAnn in target:
        newAnn = {}
        newAnn['category_id'] = objAnn['category_id']
        newAnn['label'] = categoryLabel(objAnn['category_id'], labels)
        bbox = objAnn['bbox']
        for i in range(len(bbox)):

            if i % 2 == 0:
                bbox[i] = bbox[i] / width
            else:
                bbox[i] = bbox[i] / height

        newAnn['bbox'] = bbox
        newAnn['id'] = objAnn['id']
        new_target.append(newAnn)

    return image, new_target
