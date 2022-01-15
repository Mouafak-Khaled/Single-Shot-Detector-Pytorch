import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.datasets as dset




file_path = './data/COCO/annotations/deprecated-challenge2017/labels.txt'



def readCocoLabels(file_path: str):

        labels_dict, counter = {}, 1

        labels_file = open(file_path)
        for label in labels_file:
            labels_dict[counter] = label.strip()
            counter +=1

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

    def __init__(self, rootDirectory : str,  annFile : str, split='train', transform=None):

        super(CocoData, self).__init__()

        # Make sure that the value of split is either train or validation 
        self.split = split.upper()
        assert self.split in {'TRAIN', 'VALIDATION'}

        self.transform=transform
        self.coco_dataset = None

        if self.split == 'TRAIN':

            root = os.path.join(rootDirectory, 'train2017')
            annFile = os.path.join(annFile, 'instances_train2017.json')
            self.coco_dataset = dset.CocoDetection(root = root, annFile = annFile)
        else:
            root = os.path.join(rootDirectory, 'val2017')
            annFile = os.path.join(annFile, 'instances_val2017.json')
            self.coco_dataset = dset.CocoDetection(root = root, annFile = annFile)
 
  
    def __getitem__(self, index: int):
        
        image, target = self.coco_dataset[index]
        size = transforms.ToTensor()(image).size()
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target


    def __len__(self) -> int:
        return len(self.coco_dataset)
    
    

def CustomCocoTransform(image, target):
    
    """
     - transfroms the PIL image into Pytorch tensor
     - retreive the width and hight of the image
     - resize the image to size of 300 x 300
     - generate a new annotation for the image
     - this new annotation keeps the id, the category_id, and the bbox
    """
    labels = readCocoLabels(file_path)
    
    ToTensor = transforms.ToTensor()
    image = ToTensor(image)

    height, width = image.shape[1], image.shape[2]
    new_size = [300, 300]

    Resize = transforms.Resize(new_size)
    image = Resize(image)

    new_target = []

    for objAnn in target:
        newAnn = {}
        newAnn['category_id'] = objAnn['category_id']
        newAnn['label'] = categoryLabel(objAnn['category_id'], labels)
        bbox = objAnn['bbox']
        for i in range(len(bbox)):

            if i % 2 == 0:
                bbox[i] = bbox[i] * new_size[0] / height
            else:
                bbox[i] = bbox[i] * new_size[1] / width

        newAnn['bbox'] = bbox
        newAnn['id'] = objAnn['id']
        new_target.append(newAnn)

    return image , new_target


