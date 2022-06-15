import torch
import torchvision
from data_aug.bbox_util import draw_rect
import json

def image_to_patches(obj, patch_size = (570,580), stride=(450,500)):
    images = obj[0][0]
    targets = obj[1][0]
    
    bbox = targets["boxes"].detach().clone()
    labels = targets["labels"]
    image_id = targets["image_id"]
    iscrowd =targets["iscrowd"]

    channel, WIDTH, HEIGHT = images.shape
    b = torch.arange(0,HEIGHT).unsqueeze(0).repeat(WIDTH,1).unsqueeze(0)
    a = torch.arange(0,WIDTH).unsqueeze(1).repeat(1,HEIGHT).unsqueeze(0)

    edges = torch.cat([a,b],0)
    edges = edges.unfold(1,patch_size[0],stride[1]).unfold(2,patch_size[1],stride[0]).permute(1,2,0,3,4)

    images = torch.squeeze(images.unfold(1,patch_size[0],stride[1]).unfold(2,patch_size[1],stride[0]).permute(1,2,0,3,4))
    edges = torch.flatten(edges[:,:,:,(0,-1),(0,-1)],start_dim=2).view(-1,4).tolist()
    threshold_area = 80 #(WIDTH*HEIGHT)/200
    imgs = images.detach().clone().reshape(-1,channel,patch_size[0],patch_size[1])

    target = {}
    for i,edge in enumerate(edges):
        ymin,ymax,xmin,xmax =  edge
        Xmin,Ymin,Xmax,Ymax = (bbox[:,0].clip(xmin,xmax)-xmin),(bbox[:,1].clip(ymin,ymax)-ymin),(bbox[:,2].clip(xmin,xmax)-xmin),(bbox[:,3].clip(ymin,ymax)-ymin)
        width = Xmax-Xmin
        height = Ymax - Ymin
        prob = torch.where((width*height)>threshold_area,True,False)
        area = (width*height)[prob]
        label = labels[prob]
        iscrow = iscrowd[prob]
        boxes = torch.stack((Xmin,Ymin,Xmax,Ymax),dim=-1)[prob]
        target[i] = {"boxes":boxes,"labels":label,"image_id":image_id,"area":area,"iscrowd":iscrow}
        draw_rect(imgs[i],boxes,label,f"{i}a")

    targets = [target[t] for t in range(len(target))]
    images = [i for i in imgs]

    return images,targets,edges

def patches_to_obj(images,targets,edges,iou_threshold=0.4, count=121, patch_size = (540,540), stride=(450,450)):

    images = images[0]
    for target,edge in zip(targets,edges):
        ymin,ymax,xmin,xmax = edge
        target["boxes"] = target["boxes"] +torch.tensor((xmin,ymin,xmin,ymin))

    targ = {}
    for k,t in enumerate(targets):
        if k>=len(targets)-1:
            continue
        for i, v in t.items():
            targets[k+1][i] = torch.cat((targets[k+1][i],v))

    targ = targets[-1]
    result = torchvision.ops.nms(targ["boxes"], targ["scores"], iou_threshold=iou_threshold)
    boxes = torch.index_select(targ["boxes"],0,result)
    labels = torch.index_select(targ["labels"],0,result)
    scores = torch.index_select(targ["scores"],0,result)
    draw_rect(images,boxes,labels,count)
    count+=1
    target = {"boxes":boxes,"labels":labels,"scores":scores}
    return [target]
