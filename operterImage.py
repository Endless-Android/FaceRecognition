import os


DATASET_DIR = "jm/"

count = 12
for i in range(2,16):
    img_list = os.listdir(DATASET_DIR+str(i))
    for j in range(11):
        os.rename(DATASET_DIR+str(i)+'/'+img_list[j],DATASET_DIR+'s'+str(count)+'.bmp')
        count+=1

    
    
