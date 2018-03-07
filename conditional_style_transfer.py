# Copyright (c) 2016-2017 Shafeen Tejani. Released under GPLv3.

import vgg_network

import tensorflow as tf
import numpy as np
import utils
import random
from sys import stdout
import transform

class LossCalculator:

    def __init__(self, vgg, stylized_image):
        self.vgg = vgg
        self.transform_loss_net = vgg.net(vgg.preprocess(stylized_image))
    
    def content_loss(self, content_input_batch, content_layer, content_weight,mask,size):
        content_loss =0
        content_loss_net = self.vgg.net(self.vgg.preprocess(content_input_batch))
        
        content_image1 = content_loss_net[content_layer] * mask
        print(content_image1.shape)
        generate_image1 = self.transform_loss_net[content_layer] * mask
        content_loss1 = content_weight * (2 * tf.nn.l2_loss( content_image1- generate_image1 ) / (tf.to_float(size) *128))
        
        mask = -mask+1
        content_image2 = content_loss_net[content_layer] * mask
        generate_image2 = self.transform_loss_net[content_layer] * mask
        size = 64*64 -size
        content_loss2 = 0.5*content_weight * (2 * tf.nn.l2_loss( content_image2- generate_image2 ) /( tf.to_float(size)*128))
    
        content_loss = content_loss1+ content_loss2
        return content_loss
        
    def style_loss(self, style_image, style_layers, style_weight, index):
        style_image_placeholder = tf.placeholder('float', shape=style_image.shape)
        style_loss_net = self.vgg.net(style_image_placeholder)

        with tf.Session() as sess:
            style_loss = 0
            style_preprocessed = self.vgg.preprocess(style_image)

            for layer in style_layers:
                style_image_gram = self._calculate_style_gram_matrix_for(style_loss_net,
                                                                   style_image_placeholder,
                                                                   layer,
                                                                   style_preprocessed)

                input_image_gram = self._calculate_input_gram_matrix_for(self.transform_loss_net, layer,index)

                style_loss += (2 * tf.nn.l2_loss(input_image_gram - style_image_gram) / style_image_gram.size)

            return style_weight * (style_loss)

    def tv_loss(self, image, tv_weight):
        # total variation denoising
        shape = tuple(image.get_shape().as_list())
        tv_y_size = _tensor_size(image[:,1:,:,:])
        tv_x_size = _tensor_size(image[:,:,1:,:])
        tv_loss = tv_weight * 2 * (
                (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /
                    tv_y_size) +
                (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /
                    tv_x_size))

        return tv_loss

    def _calculate_style_gram_matrix_for(self, network, image, layer, style_image):
        image_feature = network[layer].eval(feed_dict={image: style_image})
        image_feature = np.reshape(image_feature, (-1, image_feature.shape[3]))
        return np.matmul(image_feature.T, image_feature) / image_feature.size

    def _calculate_input_gram_matrix_for(self, network, layer,index):
        image_feature = network[layer][:,index[0][0]:index[2][0], index[1][0]:index[3][0],:]
        print(image_feature.shape)
        batch_size, height, width, number = map(lambda i: i.value, image_feature.get_shape())
        height = index[2]-index[0]
        width = index[3]- index[1]
        image_feature = tf.reshape(image_feature, (batch_size, -1, number))
        size  = height*width*number
        return tf.matmul(tf.transpose(image_feature, perm=[0,2,1]), image_feature) / tf.to_float(size)



class ConditionalStyleTransfer:
    CONTENT_LAYER = 'relu3_3'
    STYLE_LAYERS = [ 'relu3_3']

    def __init__(self, vgg_path,
                style_image, content_shape, content_weight,
                style_weight, tv_weight, batch_size, device):
        
            vgg = vgg_network.VGG(vgg_path)
            self.style_image = style_image
            self.batch_size = batch_size
            self.batch_shape = (batch_size,) + content_shape

            self.input_batch = tf.placeholder(tf.float32,
                                              shape=self.batch_shape,
                                              name="input_batch")
            
            self.net_mask_batch = tf.placeholder(tf.float32, shape = [4,64,64,1])
            self.index = tf.placeholder(tf.int32, shape=[4,1])
            self.size = tf.placeholder(tf.int32,shape=[1])
            
            self.stylized_image = transform.net(self.input_batch,self.net_mask_batch)

            loss_calculator = LossCalculator(vgg, self.stylized_image)

            self.content_loss = loss_calculator.content_loss(
                                            self.input_batch,
                                            self.CONTENT_LAYER,
                                            content_weight,
                                            self.net_mask_batch,
                                            self.size) / self.batch_size

            self.style_loss = loss_calculator.style_loss(
                                            self.style_image,
                                            self.STYLE_LAYERS,
                                            style_weight,
                                            self.index) / self.batch_size

            self.total_variation_loss = loss_calculator.tv_loss(
                                            self.stylized_image,
                                            tv_weight) / batch_size

            self.loss = self.content_loss  + self.style_loss + 0*self.total_variation_loss
            
            
    def get_mask(self,batch_size):
        mask_array = np.zeros([batch_size,64,64,1])
    
        mask_array2 = np.zeros([batch_size,256,256,1])
    
        x0= random.sample(range(110),1)[0]/4
        x1 = random.sample(range(130,256),1)[0]/4
        y0= random.sample(range(110),1)[0]/4
        y1= random.sample(range(130,256),1)[0]/4

        p = random.random()
        if p >0.9:
            x0=0
            y0=0
            x1=63
            y1=63
        mask_array[:,x0:x1,y0:y1,0]=1 
        mask_array2[:,x0*4:x1*4,y0*4:y1*4,0]=1 

        index = np.array([x0,y0,x1,y1])
        index = np.expand_dims(index,1)
        size=np.count_nonzero(mask_array[0]==1)
        size = np.array([size])
        return mask_array,mask_array2, index ,size


    def _current_loss(self, feed_dict):
        losses = {}
        losses['content'] = self.content_loss.eval(feed_dict=feed_dict)
        losses['style'] = self.style_loss.eval(feed_dict=feed_dict)
        losses['total_variation'] = self.total_variation_loss.eval(feed_dict=feed_dict)
        losses['total'] = self.loss.eval(feed_dict=feed_dict)
        return losses

    def train(self, content_training_images,learning_rate,
              epochs, checkpoint_iterations):

        def is_checkpoint_iteration(i):
            return (checkpoint_iterations and i % checkpoint_iterations == 0)

        def print_progress(i):
            stdout.write('Iteration %d\n' % (i + 1))

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            iterations = 0
            for epoch in range(epochs):
                for i in range(0, len(content_training_images)-3, self.batch_size):
                    

                    batch = self._load_batch(content_training_images[i: i+self.batch_size])
                    mask,mask2,index,size =self.get_mask(4)
                    train_step.run(feed_dict={self.input_batch:batch, self.net_mask_batch : mask, self.index :index,self.size : size})
                    
                    if i %100 ==0 :
                        print_progress(iterations)
                        self.output = sess.run(self.stylized_image,feed_dict={self.input_batch:batch,self.net_mask_batch : mask, self.index :index,self.size : size})
                        utils.save_image(self.output[0],'result.jpg')
                    
                    if is_checkpoint_iteration(iterations):
                        yield (
                            iterations,
                            sess,
                            self.stylized_image.eval(feed_dict={self.input_batch:batch,self.net_mask_batch : mask, self.index :index,self.size : size})[0],
                            self._current_loss({self.input_batch:batch,self.net_mask_batch : mask, self.index :index,self.size : size})
                       )
                    iterations += 1

    def _load_batch(self, image_paths):
        return np.array([utils.load_image(img_path,[256,256]) for j, img_path in enumerate(image_paths)])
        



def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)
