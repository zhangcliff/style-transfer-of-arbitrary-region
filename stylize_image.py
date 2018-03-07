
import numpy as np
from os.path import exists
from sys import stdout
from skimage.io import imread
import utils
from argparse import ArgumentParser
import tensorflow as tf
import transform


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content', type=str,
                        dest='content', help='content image path',
                        metavar='CONTENT', required=True)

    parser.add_argument('--mask', type=str,
                        dest='mask',
                        help='mask image path',
                        metavar='Mask', required= True)

    parser.add_argument('--output-path', type=str,
                        dest='output_path',
                        help='path for output',
                        metavar='OUTPUT_PATH', required=True)
    
    parser.add_argument('--model-path', type=str,
                        dest='model_path',
                        help='path for model',
                        metavar='MODEL_PATH', required=True)
    return parser

def check_opts(opts):
    assert exists(opts.content), "content not found!"
    assert exists(opts.mask), "mask not found!"


def main():
    parser = build_parser()
    options = parser.parse_args()
    check_opts(options)
    
    shape = imread(options.content).shape
    content_image = utils.load_image(options.content,[256,256])
    content_image = np.expand_dims(content_image,axis=0)
    
    mask_image = utils.load_image(options.mask,[64,64])[...,0:1] /255.
    mask_image = np.expand_dims(mask_image,0)
    
    model = options.model_path
    prediction = ffwd(content_image,mask_image,model)
    utils.save_image(prediction,[shape[0],shape[1]], options.output_path)


def ffwd(content,mask,model):
    with tf.Session() as sess:
        shape=list(content.shape)
        img_placeholder = tf.placeholder(tf.float32, shape=shape,
                                         name='img_placeholder')
        print(img_placeholder.shape)
        mask_placeholder = tf.placeholder(tf.float32,shape=[1,64,64,1])
        network = transform.net(img_placeholder,mask_placeholder)
        saver = tf.train.Saver()

        saver.restore(sess, model)

        prediction = sess.run(network, feed_dict={img_placeholder:content,mask_placeholder :mask})
        return prediction[0]

if __name__ == '__main__':
    main()
