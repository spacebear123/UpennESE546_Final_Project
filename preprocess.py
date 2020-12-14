print("start")
import pdb 
import os
import sys
import random
import math
import numpy as np
import scipy
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import traceback
from myUtils import print_mask, print_mask_val

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn import config

ROOT_DIR = os.path.abspath("./Mask_RCNN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
print(COCO_MODEL_PATH)
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class HorseConfig(config.Config):
     NAME = "horse"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 80
    
horseConfig = HorseConfig()
horseConfig.display()


model = modellib.MaskRCNN(mode="inference", model_dir=COCO_MODEL_PATH, config=horseConfig)
model.load_weights(COCO_MODEL_PATH,by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
def preprocess(img_dirs, selected_class="horse"):
    bad_list = defaultdict(lambda: defaultdict(list))
    for img_dir in img_dirs:
        images = [skimage.io.imread]
        base_img_dir = os.path.basename(img_dir)
        output_dir = os.path.join(os.path.dirname(img_dir), "masked_"+base_img_dir)
        os.makedirs(output_dir, exist_ok=True)
        print("making", output_dir, "from", img_dir)
        sys.stdout.flush()
        files = next(os.walk(img_dir))[2]
        for f_cnt, file_name in enumerate(files):
            # (H, W, C=3)
            try:
                image = skimage.io.imread(os.path.join(img_dir, file_name))
                print("image: %s"%file_name)
                res = model.detect([image])[0]
                masks = res['masks'] # (H, W, M)
                class_ids = res['class_ids']
                preds = [class_names[cls_id] for cls_id in class_ids if class_names[cls_id] == selected_class]
                print(preds)
                if not preds:
                    print("Warning: no %s detected in "%selected_class, file_name)
                    bad_list['no'][base_img_dir].append(file_name)
                    continue
                pred_cnt = Counter(preds)
                mask_idxs = [idx for idx in range(masks.shape[2]) if class_names[class_ids[idx]] == selected_class]

                hz_masks = masks[:, :, mask_idxs]

                mask = np.logical_or.reduce(hz_masks, axis=2, keepdims=True)
                map_fn = np.vectorize(lambda x: 255 if x else 0)
                mask = map_fn(mask)

                catted = np.concatenate([image, mask.astype(np.int32)], axis=2)
                file_name_no_ext = os.path.splitext(file_name)[0]
                out_file_name = ".".join([file_name_no_ext, "npy"])
                scipy.misc.toimage(catted, cmin=0.0, cmax=255.).save(file_name_no_ext+'.png')
                print_mask_val(catted[:,:,3])
            except Exception as e:
                print("Exception when handling image %s!"%file_name)
                traceback.print_exc()
        sys.stdout.flush()

    bad_list = dict(bad_list)
    for k, v in bad_list.items():
        bad_list[k] = dict(v)
    np.save('bad_list.npy', dict(bad_list))
    print("Images with warnings:")
    print(bad_list)



HORSE_DIRS = ["images"]

if __name__ == "__main__":

    preprocess(HORSE_DIRS, "horse")
