import sys
import os
sys.path.append("../img2vec_pytorch")  # Adds higher directory to python modules path.
from img_to_vec import Img2Vec

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import cv2

input_path = './test_images' # './grayscale', './sketch_kernel15'
image_format = 'rgb' #'grayscale' 'sketch'
save_converted_image = True

print("Getting vectors for test images...\n")
# img2vec = Img2Vec()
img2vec = Img2Vec(model='vgg')

# A function that returns the number of the string:
def myFunc(file):
  return int(os.fsdecode(file).split('.')[0])

def generate_matchDict(imgNum):
    matchDict = {}
    for i in range(imgNum):
        if i == 12:
            matchDict[i] = [13, 14]
        elif i == 13:
            matchDict[i] = [12, 14]
        elif i == 14:
            matchDict[i] = [12, 13]
        elif 14 < i < 45:
            if i % 2 == 0:
                matchDict[i] = [i-1]
            else:
                matchDict[i] = [i+1]
        elif i == 45:
            matchDict[i] = [46, 47]
        elif i == 46:
            matchDict[i] = [45, 47]
        elif i == 47:
            matchDict[i] = [45, 46]
        else:
            if i % 2 == 0:
                matchDict[i] = [i+1]
            else:
                matchDict[i] = [i-1]
    return matchDict

matchDict = generate_matchDict(66) # Current dataset has 66 images

def correctMatch(numQuery, numList):
    print(numQuery, numList)
    if numQuery == 12:
        return True if 13 in numList or 14 in numList else False
    elif numQuery == 13:
        return True if 12 in numList or 14 in numList else False
    elif numQuery == 14:
        return True if 12 in numList or 13 in numList else False
    elif 14 < numQuery < 45:
        if numQuery % 2 == 0:
            return True if (numQuery - 1) in numList else False
        else:
            return True if (numQuery + 1) in numList else False
    elif numQuery == 45:
        return True if 46 in numList or 47 in numList else False
    elif numQuery == 46:
        return True if 45 in numList or 47 in numList else False
    elif numQuery == 47:
        return True if 45 in numList or 46 in numList else False
    else:
        if numQuery % 2 == 0:
            return True if (numQuery + 1) in numList else False
        else:
            return True if (numQuery - 1) in numList else False

def img2sketch(photo, k_size, save_path):
    #Read Image
    img=cv2.imread(photo)
    
    # Convert to Grey Image
    grey_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert Image
    invert_img=cv2.bitwise_not(grey_img)
    #invert_img=255-grey_img

    # Blur image
    blur_img=cv2.GaussianBlur(invert_img, (k_size,k_size),0)

    # Invert Blurred Image
    invblur_img=cv2.bitwise_not(blur_img)
    #invblur_img=255-blur_img

    # Sketch Image
    sketch_img=cv2.divide(grey_img,invblur_img, scale=256.0)

    # Save Sketch 
    cv2.imwrite(save_path, sketch_img)

    # # Display sketch
    # cv2.imshow('sketch image',sketch_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# For each test image, we store the filename and vector as key, value in a dictionary
pics = {}
fileList = sorted(os.listdir(input_path), key = myFunc)
kernel_size = 15
path_grayscale = './grayscale'
path_sketch = f'./sketch_kernel{kernel_size}'
os.makedirs(path_grayscale, exist_ok=True)
os.makedirs(path_sketch, exist_ok=True)

for file in fileList:
    filename = os.fsdecode(file)
    file_path = os.path.join(input_path, filename)
    if image_format == 'rgb':
        img = Image.open(file_path).convert('RGB')
    else:
        img = Image.open(file_path).convert('L').convert('RGB') # Convert to grayscale
    if save_converted_image:
        if image_format == 'grayscale':
            imgsave = img.save(f"./grayscale/{file}")
        elif image_format == 'sketch':
            img2sketch(photo = file_path, k_size=kernel_size, save_path = f"{path_sketch}/{file}")
    vec = img2vec.get_vec(img)
    pics[filename] = vec

available_filenames = ", ".join(pics.keys())
pic_name = ""
correct = 0
rankingList =[]

for pic_name in list(pics.keys()):
    ind = int(pic_name.split('.')[0])
    sims = {}
    for key in list(pics.keys()):
        if key == pic_name:
            continue

        sims[key] = cosine_similarity(pics[pic_name].reshape((1, -1)), pics[key].reshape((1, -1)))[0][0]

    d_view = [(v, k) for k, v in sims.items()]
    d_view.sort(reverse=True)
    i = 0
    subList = []
    with open('similarity_results.txt', 'a') as f:
        f.write(pic_name)
        f.write('\n')

    for _i, (v, k) in enumerate(d_view):
        currentInd = int(k.split('.')[0])
        if currentInd in matchDict[ind]:
            rankingList.append(_i)

        with open('similarity_results.txt', 'a') as f:
            f.write(', '.join((str(v), str(k))))
            f.write('\n')
        

        i += 1
        subList.append(currentInd)
        if i == 5:
            # with open('similarity_results.txt', 'a') as f:
            #     f.write('\n')      
            correct = correct + correctMatch(ind, subList)
            # break
    with open('similarity_results.txt', 'a') as f:
        f.write('\n')

with open('similarity_results.txt', 'a') as f:
    f.write(f'Correct num = {correct}\n')
    avegrageRank = float(sum(rankingList)/len(rankingList))
    f.write(f'Average rankings = {avegrageRank}')
        

