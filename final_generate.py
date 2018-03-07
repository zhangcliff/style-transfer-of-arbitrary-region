# -*- coding: utf-8 -*-
"""
Created on Thu Dec 07 20:56:25 2017

@author: Administrator
"""

import numpy as np
from argparse import ArgumentParser
from skimage.io import imread,imshow,imsave


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content', type=str,
                        dest='content', help='content image path',
                        metavar='CONTENT', required=True)

    parser.add_argument('--mask', type=str,
                        dest='mask',
                        help='mask image path',
                        metavar='Mask', required= True)

    parser.add_argument('--stylized', type=str,
                        dest='stylized',
                        help='path for stylized image',
                        metavar='stylized_PATH', required=True)
    
    parser.add_argument('--output-path', type=str,
                        dest='output_path',
                        help='path for output',
                        metavar='output_PATH', required=True)
    return parser


def generate_final():
    parser = build_parser()
    options = parser.parse_args()
    
    yc= imread(options.content)
    mask = imread(options.mask)/220
    mask = np.expand_dims(mask,axis=2)
    yg= imread(options.stylized)

    yf=yg*mask+yc*(-mask+1)

    imsave(options.output_path,yf)
    
    
if __name__ == '__main__':
    generate_final()
    