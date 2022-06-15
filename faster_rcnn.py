import json
import random

from data_aug.bbox_util import draw_rect

def loadAnns(anns, ids=[]):
    """
    Load anns with the specified ids.
    :param ids (int array)       : integer ids specifying anns
    :return: anns (object array) : loaded ann objects
    """
    if isinstance(ids,list):
        return [anns[id] for id in ids]
    elif type(ids) == int:
        return [x for x in anns if x["image_id"]==ids]

def loadImgs(anns, ids=[]):
    """
    Load anns with the specified ids.
    :param ids (int array)       : integer ids specifying img
    :return: imgs (object array) : loaded img objects
    """
    if isinstance(ids,list):
        return [anns[id] for id in ids]
    elif type(ids) == int:
        return [x for x in anns if x["id"]==ids]
        
def coco_split(annotations,split=0.8):
#   Splits COCO annotations file into training and test sets.

# positional parameters:
#   coco_annotations      Path to COCO annotations file.

# optional parameters:
#   split                 A percentage of a split; a number in (0, 1)
#   having-annotations    Ignore all images without annotations. Keep only these
#                         with at least one annotation
#   multi-class           Split a multi-class dataset while preserving class
#                         distributions in train and test sets

    with open(annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco['info']
        licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']


        test_images = [img for img in images if random.random()>split]
        test_img_ids = [img["id"] for img in test_images]
        test_annotations = [ann for ann in annotations if ann["image_id"] in test_img_ids]

        train_images = [img for img in images if img["id"] not in test_img_ids]
        train_img_ids = [img["id"] for img in train_images]
        train_annotations = [ann for ann in annotations if ann["image_id"] in train_img_ids]

        train = {"info":info,"licenses":licenses,"images":train_images,"annotations":train_annotations,"categories":categories}
        print("train_len", len(train_images),"train_ann_len",len(train_annotations))
        if split != 1:
          test =  {"info":info,"licenses":licenses,"images":test_images,"annotations":test_annotations,"categories":categories}
        else:
          print("test verisi ayrılmadı")
          return train
        print("test_len", len(test_images),"test_ann_len",len(test_annotations))

        return train, test

import os
import torch
print("pytorch version:",torch.__version__)
torch.manual_seed(1)
from torch import nn
from torchvision import datasets, models
from vision2.engine import train_one_epoch, evaluate
from torch.utils.data import DataLoader,Dataset

import copy
import cv2
from albumentations.pytorch import ToTensorV2
import albumentations as A  # our data augmentation library

import warnings
warnings.filterwarnings("ignore")

def get_transforms(train=False):
    if train:
        transform = A.Compose([
            A.RandomCrop(570,580),
            A.Rotate(int(torch.rand((1,))*7),p=1.0,),
            A.HorizontalFlip(),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc',label_fields=["labels","area","iscrowd"],min_visibility=0.2,min_area=30))
    else:
        transform = A.Compose([
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc',label_fields=["labels","area","iscrowd"]))
    return transform

class Dataset(datasets.VisionDataset):
    def __init__(self, coco_dir, data_dir, transform=None, target_transform=None, transforms=None):
        # the 3 transform parameters are reuqired for datasets.VisionDataset
        super().__init__(data_dir, transforms, transform, target_transform)
        self.coco_dir = coco_dir
        self.data_dir = data_dir
        self.categories = [i["name"] for i in coco_dir["categories"]]

        if isinstance(self.coco_dir,dict):
          self.ids = [x["id"] for x in coco_dir["images"] if len(self._load_target(x["id"]))>0]
    
    def _load_image(self, id: int):
        name = loadImgs(self.coco_dir["images"],id)[0]['file_name']
        image = cv2.imread(os.path.join(self.data_dir, name))
        image = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return image.div(255).numpy()

    def _load_target(self, id):
        return loadAnns(self.coco_dir["annotations"],id)
    
    def n_classes(self):
        #load classes
        print(self.categories)
        return ["__background__"]+[i for i in self.categories]
        
    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        # target = self._load_target(id)
        target = copy.deepcopy(self._load_target(id))
        boxes = torch.tensor([t["bbox"] for t in target]) # required annotation format for albumentations
        labels = torch.tensor([t["category_id"] for t in target], dtype=torch.int64)
        iscrowd = torch.tensor([t["iscrowd"] for t in target], dtype=torch.bool)
        area = torch.multiply(boxes[:,2],boxes[:,3])#torch.multiply((boxes[:,2]-boxes[:,0]),(boxes[:,3]-boxes[:,1])) #

        new_boxes = torch.add(boxes[:,:2],boxes[:,2:])
        boxes = torch.cat((boxes[:,:2],new_boxes),-1)

        transformed = self.transforms(image=image, bboxes=boxes,area=area,labels=labels,iscrowd=iscrowd)
        image = transformed['image']
        boxes = torch.tensor(transformed["bboxes"],dtype=torch.int64).view(-1,4)
        labels = torch.tensor(transformed["labels"],dtype=torch.int64).view(-1,)
        iscrowd = torch.tensor(transformed["iscrowd"],dtype=torch.bool).view(-1,)
        area = torch.tensor(transformed["area"]).view(-1,)

        targ = {} # here is our transformed target
        targ['boxes'] = boxes
        targ['labels'] = labels
        targ['image_id'] = torch.tensor([id])
        targ['area'] = area
        targ['iscrowd'] = iscrowd

        return image, targ # scale images

    def __len__(self):
        return len(self.ids)

def train(model,data_loader_train,data_loader_test,num_epochs,classes=None, lr=0.0002, momentum=0.9, weight_decay=0.0005,step_size=3,gamma=0.5,print_freq=10):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # model = model.to(device)
    model = get_model(18)

    for param in model.parameters():
        param.required_grad = False

    model.load_state_dict(torch.load("/home/berat/Desktop/faster_rcnn/dataset/visdrone/models/model3.pth"),strict=False)

    params = [p for p in model.parameters() if p.requires_grad]


    optimizer = torch.optim.Adam(params, 
                                lr=lr,
                                weight_decay=weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma) 
    
    for epoch in range(num_epochs):
        train_one_epoch(model, 
                       optimizer, 
                       data_loader_train, 
                       device, 
                       epoch, 
                       print_freq=print_freq)
        # lr_scheduler.step()

        if data_loader_test != None:
            evaluate(model, data_loader_test, device=device)

        torch.save(model.state_dict(), 
                os.path.join(dir,"models",f"model{epoch}.pth"))

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(n_classes):
    # lets load the faster rcnn model
    model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)
    return model

def main(dir,num_epochs):
    dataset_dir = os.path.join(dir,"images")
    dataset_ann = os.path.join(dir,"coco/annotations.json")

    train_ann,test_ann = coco_split(dataset_ann, split= 0.90) # split 1 ken tüm veri train'e atılır.

    train_dataset = Dataset(train_ann,dataset_dir, transforms=get_transforms(True))
    test_dataset = Dataset(test_ann,dataset_dir,transforms=get_transforms(False))
    
    train_loader = DataLoader(train_dataset, batch_size=15, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)

    classes = train_dataset.n_classes()
    model = get_model(len(classes))

    train(model,train_loader,test_loader,num_epochs,classes=classes)

if __name__== "__main__":
    dir = "/home/berat/Desktop/faster_rcnn/dataset/visdrone"

    main(dir,num_epochs = 30)