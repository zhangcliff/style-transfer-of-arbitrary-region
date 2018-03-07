# style-transfer-of-arbitrary-region

This is the code for the paper:
Local Perceptual Loss Function for Arbitrary Region Style Transfer (ICPR2018 under  reviewing)

## About
This is a tensorflow implementation of a conditional style transfer network which is trained to realize style transfer of arbitrary region. The conditional transfer network has two inputs, one is content image, the other one is a mask image. For example, if we want to change the left half part pf the content image, we only need to feed the content image and the correspodding mask to the model. The result is shown as 
![image](https://github.com/zhangcliff/style-transfer-of-arbitrary-region/blob/master/example/example1.png)

## Usage
You can download all trained models from here [Baidu Yun](https://pan.baidu.com/s/16YkBPWW_9jQj8-Qa4zedsQ)
```shell
python stylize_image.py --content <content image> --mask <mask image> --model-path <model-path> --output-path <output image path>
```
For example:
```shell
python stylize_image.py --content content/dog.jpg  --mask mask/mask1.jpg  --model-path logfile_night/night.ckpt --output-path stylized_image.jpg
```
## Train
The content image can be downloaded from [coco](http://msvocds.blob.core.windows.net/coco2014/train2014.zip).
The pre-trained vgg19 modelfile can be downloaded from [VGG19](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat).
Then you can train a new model as
```shell
python train_network.py --style <style_image> --train-path <the content image folder> --save-path <the save path> --vgg-path <the pretrained vgg19 model>
```
for example
```shell
python train_network.py --style style/style1.jpg --train-path train2014 --save-path logfile --vgg-path imagenet-vgg-verydeep-19.mat
```

## Sample results
![image](https://github.com/zhangcliff/style-transfer-of-arbitrary-region/blob/master/example/stylized_image.png)

When look at the above results,we will find that there are some defect at the edge of the stylized picture.Therefore we can combine the origin content image with the stylized picture to generate more beautiful one. You can run 

The finnal results are as follow:
![image](https://github.com/zhangcliff/style-transfer-of-arbitrary-region/blob/master/example/final_image.png)

