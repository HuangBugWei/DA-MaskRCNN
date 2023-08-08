import os, cv2
import numpy as np
from PIL import Image

dir1 = './ablation-DA-25k/0059999/0.5'
dir2 = './ablation-no-DA-25k/0059999/0.5'
dir3 = './ablation-vanilla-235/0059999/0.5'

outputPath = './ablation-result'
os.makedirs(outputPath, exist_ok=True)

def process(dirname, filename):
    image = cv2.imread(os.path.join(dirname, filename))
    tag = np.zeros((50, image.shape[1], 3), np.uint8)
    cv2.putText(tag, dirname, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate((tag, image), axis = 0)

for filename in os.listdir(dir1):
    print(filename)
    image1 = process(dir1, filename)
    image2 = process(dir2, filename)
    image3 = process(dir3, filename)
    
    mix = np.concatenate((image1, image2, image3), axis=0)
    cv2.imwrite(os.path.join(outputPath, filename), mix)
