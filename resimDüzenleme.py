from PIL import Image
import glob
import torch
from torchvision import transforms

ct = transforms.ToTensor()
image_list = []
count = 0
# Open a file with access mode 'a'
file_object = open('small_788.txt', 'a')

for filename in glob.glob('/home/berat/Desktop/faster_rcnn/dataset/images/*.jpg'): #assuming gif
    im=ct(Image.open(filename))
    # Append 'hello' at the end of file

    if im.shape[1]<800:
        file_object.write(".".join("/".join(filename.split("/")[-1]).split(".")[:-1])+".txt\n")
        # os.rename(filename,filename.replace("images","silinenler"))
        # txt = filename.replace("jpg","txt").replace("dataset/images","annotations")
        # os.rename(txt,txt.replace("annotations","dataset/silinenler"))

        print(filename,im.shape)
        count += 1
print(count)
# Close the file
file_object.close()