import argparse

import os
os.environ["GLOG_minloglevel"] = "1"
import caffe
import sys
sys.path.append('models/research/slim')
import tensorflow as tf
from nets.mobilenet import mobilenet_v2

parser = argparse.ArgumentParser()
parser.add_argument('--image_scale', type=int, default=224,
                    help='specify the input image scale.')
parser.add_argument('--factor', type=float, default=1.0,
                    help='factor for mobilenet-v2')
args = parser.parse_args()


def generator(image_scale, factor, eps=1e-3):
    ### load tf model
    base_name = '_'.join(['mobilenet_v2', str(factor), str(image_scale)])  # ex: 'mobilenet_v2_1.0_224'
    checkpoint = base_name + '/' + base_name + '.ckpt'

    print('Building tf graph ...')
    tf.reset_default_graph()
    images = tf.placeholder(tf.float32, shape=(None, image_scale, image_scale, 3))
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
        logits, endpoints = mobilenet_v2.mobilenet(images, num_classes=1001, depth_multiplier=factor,
                                                   finegrain_classification_mode=True)
    ema = tf.train.ExponentialMovingAverage(0.999)
    vars = ema.variables_to_restore()
    saver = tf.train.Saver(vars)

    print('Parsing tf variables ...')
    ### convert variables from tf checkpoints to caffemodel
    channel_list = []
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        tf_all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        i = 0
        for var in tf_all_vars:
            variable_shape = var.shape.as_list()
            if len(variable_shape) == 4:
                if variable_shape[-1] == 1:
                    n_channel = variable_shape[-2]
                else:
                    n_channel = variable_shape[-1]
                channel_list.append(n_channel)
                i+=1
    ### Numbers of channels are stored in channel_list, with length = 53
    channel_list[-1] = 1000 # 1000 classes
    print('channels:', channel_list)
    print('------------------------------------------------------------------')

    ### read prototxt_mobilenet_v2/mobilenet_v2_1.0_224.prototxt,
    ### and change "num_output" and "group" in convolution_param
    with open('prototxt_mobilenet_v2/mobilenet_v2_1.0_224.prototxt', 'r') as f:
        lines = f.readlines()
    with open('prototxt_mobilenet_v2/' + base_name + '.prototxt', 'w') as g:
        i = 0
        for line in lines:
            # change image scale
            if 'input_dim: 224' in line:
                line = str(image_scale).join([line[:line.index('224')], '\n'])
            # change num_output
            if 'num_output:' in line:
                line = str(channel_list[i]).join([line[:line.index(':') + 2], '\n'])
                i+=1
            # change group
            if 'group:' in line:
                line = str(channel_list[i-1]).join([line[:line.index(':') + 2], '\n'])
            g.write(line)
    print('Successfully generate protxt:', base_name+'.prototxt')
    return


if __name__ == '__main__':
    generator(args.image_scale, args.factor, eps=1e-3)