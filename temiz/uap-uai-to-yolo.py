import os         
import numpy as np 
os.chdir("/home/berat/Masaüstü/uap-uai/annotations")
print(os.getcwd()) 

data_folder = os.path.join(os.getcwd())
a=0
data=[]

for root, folders, files in os.walk(data_folder):
    
    for file in files:
        #print(file)
        splittedlist = []
        mynewliste = []
        newstr = ""
        strliste = []
        path = os.path.join(root,file) 
        with open(path, 'a+') as dosya:
             dosya.seek(0)
             for line in dosya:
                 splittedlist.append(line.strip("\n").split(" ")) 

        mynewliste = np.array([[float(k) for k in i] for i in splittedlist])
        
        # for i in range(len(mynewliste)):
        #     mynewliste[]
        print((mynewliste[0][1]))
        """
                    xmin = (xcenter*w_img)-w/2
                    ymin = (ycenter*h_img)-h/2
                    width = w*w_img 1920
                    height = h*h_img 1080
        """

            
        # strliste = np.array_str(mynewliste)
        # newstr = strliste.replace(" ","") 
        # newstr = newstr.replace("''",",") 
        # newstr = newstr.replace("['","") 
        # newstr = newstr.replace("']","") 
        # newstr = newstr.replace("[","") 
        # newstr = newstr.replace("]","") 
        #print(newstr)
    """   with  open(path,'w') as dosya:
                dosya.write(newstr)
    """
 