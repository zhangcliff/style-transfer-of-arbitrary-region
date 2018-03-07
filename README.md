# style-transfer-of-arbitrary-region

This is the code for the paper:
Local Perceptual Loss Function for Arbitrary Region Style Transfer (ICPR2018 under  reviewing)

## About
This is a tensorflow implementation of a conditional style transfer network which is trained to realize style transfer of arbitrary region. The conditional transfer network has two inputs, one is content image, the other one is a mask image. For example, if we want to change the left half part pf the content image, we only need to feed the content image and the correspodding mask to the model. The result is shown as 
![image](https://github.com/zhangcliff/style-transfer-of-arbitrary-region/blob/master/example/example1.png)

## Usage
''' python
python stylize_image.py --content <> --mask <> --model-path <> --output-path <>
'''
