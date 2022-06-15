import json
import os
from PIL import Image

img_id = 0
ann_id = 0
category_dict = {}
category_instancesonly = {
    '__background__':0 ,
    'ig':-1,
    'pedestrian':1,
    'people':1,
    'bicycle':2 ,
    'car':3,
    'van':4,
    'truck':4,
    'tricycle': 2,
    'awning-tricycle':2 ,
    'bus':4,
    'motor':2,
    'others':5,
    'uai':6,
    'uap':7,
}
category_id = {
    0: '__background__',
    1: 'ig',
    2: 'pedestrian',
    3: 'people',
    4: 'bicycle',
    5: 'car',
    6: 'van',
    7: 'truck',
    8: 'tricycle',
    9: 'awning-tricycle',
    10: 'bus',
    11: 'motor',
    12: 'others',
    13: 'uai',
    14: 'uap',
    
}
ann_dict = {}
images = []
annotations = []

data_root = '/home/berat/Desktop/faster_rcnn/dataset/visdrone'
for root, dirs, files in os.walk(os.path.join(data_root, 'annotations')):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(root, file)
            if len(images) % 100 == 0:
                print("Processed %s images, %s annotations" % (
                    len(images), len(annotations)))
            with open(file_path, "r") as f:
                try:
                    img = Image.open(
                        os.path.join(data_root, 'images', file.split('.')[0] + '.jpg')    # burası 0 döndürür ise aşağıya bakar 
                        )
                except:
                    img = Image.open(
                        os.path.join(data_root, 'images', file.split('.')[0] + '.png')
                    )
                width, height = img.size[0], img.size[1]
                image = {}
                image['id'] = img_id
                img_id += 1
                image['width'] = width
                image['height'] = height
                image['file_name'] = file.split('.')[0] + '.jpg'
                images.append(image)
                for line in f.readlines():
                    line = line.strip('\n')
                    """
                    xmin = (xcenter*w_img)-w/2
                    ymin = (ycenter*h_img)-h/2
                    width = w*w_img 1920
                    height = h*h_img 1080
                    """
                    xmin, ymin, w, h,iscrowd,cls= float(line.split(',')[0]), float(line.split(',')[1]), float(
                        line.split(',')[2]), float(line.split(',')[3]), int(line.split(',')[6]), int(line.split(',')[5])

                    cls = category_instancesonly[category_id[cls]]  

                    xmax = xmin+w
                    ymax = ymin+h
                    
                    ann = {}
                    ann['id'] = ann_id
                    ann_id += 1
                    ann['image_id'] = image['id']
                    ann['area'] = w * h
                    ann['iscrowd'] = iscrowd
                    ann['bbox'] = [xmin,ymin,xmax,ymax]

                    ann['category_id'] = cls
                    if cls != -1:
                        annotations.append(ann)

classes = ["person","small_vehicle","medium_vehicle","big_vehicles","huge_vehicles","uai","uap"]                        
ann_dict['info'] = []
ann_dict['licenses'] = []
ann_dict['images'] = images
categories = [{"id": i, "name": name} for i,name in enumerate(classes)]
ann_dict['categories'] = categories
ann_dict['annotations'] = annotations
print("Num categories: %s" % len(categories))
print("Num images: %s" % len(images))
print("Num annotations: %s" % len(annotations))
with open(os.path.join(data_root, 'coco', 'annotations.json'), 'w') as outfile:
    outfile.write(json.dumps(ann_dict))