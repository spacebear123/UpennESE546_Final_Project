Before running the Mask RCNN,
Please install the requiremetns in the requirements_mrcnn.txt exactly according to the version numbers, as tf and keras don't have proper backward compatibilities, a higher version would break the code.

Then please run 
python3 Mask_RCNN/setup.py install

Then, please download the mask_rcnn_coco.h5 file from https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 as the pretrained weights. And put it inside Mask_RCNN folder.

Then, place pictures with horses(images without horses would trigger a warning during generation) under /images folder.
Then you can start the first part of the generator by running

python3 preprocess.py

Which would generate masked images in the root folder.

Then, setup cycle-GAN according to the requirements_cycle.txt,
then run 

source cycleGan-pix2pix/datasets/download_cyclegan_dataset.sh horse2zebra

place the freshly generated pictures(You need to convert them to .JPG) from Mask RCNN into cycleGan-pix2pix/datasets/horse2zebra/testA (You can delete the predownloaded pictures if you want to)

Then, run 

python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

And you can find the generated pictures in results folder, where horses are transformed to zebras.

Lastly, run the post_process.ipynb process by placing the original image, Mask RCNN result, and final Cycle GAN under the correct folders (they are named for you), to see the final result of the project.
