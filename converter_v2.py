import numpy as np
import os
import argparse

os.environ["GLOG_minloglevel"] = "1"
import caffe
import sys
sys.path.append('models/research/slim')
import tensorflow as tf
from nets.mobilenet import mobilenet_v2


parser = argparse.ArgumentParser()
parser.add_argument('--image_scale', type=int, default=224,
                    help='input image scale.')
parser.add_argument('--factor', type=float, default=1.0,
                    help='factor for mobilenet-v2')
parser.add_argument('--image_file', type=str, default='imgs/panda.jpg',
                    help='path/to/imagefile')
args = parser.parse_args()

image_scale = args.image_scale
factor = args.factor
image_file = args.image_file

if not os.path.exists('caffemodel_fromckpt'):
    os.mkdir('caffemodel_fromckpt')


base_name = '_'.join(['mobilenet_v2', str(factor), str(image_scale)]) # ex: 'mobilenet_v2_1.0_224'
prototxt = 'prototxt_mobilenet_v2/' + base_name + '.prototxt'
checkpoint = base_name + '/' + base_name + '.ckpt'
to_caffemodel = 'caffemodel_fromckpt/' + base_name + '.caffemodel'

print(base_name)

def caffe_load_from_ckpt(prototxt, checkpoint, to_caffemodel):
    ### load caffe model and weights
    caffe.set_mode_gpu()
    net = caffe.Net(prototxt, caffe.TEST)

    ### load tf model
    tf.reset_default_graph()
    images = tf.placeholder(tf.float32, shape=(None, image_scale, image_scale, 3))
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
        logits, endpoints = mobilenet_v2.mobilenet(images, num_classes=1001, depth_multiplier=factor, finegrain_classification_mode=True)
    ema = tf.train.ExponentialMovingAverage(0.999)
    vars = ema.variables_to_restore()
    saver = tf.train.Saver(vars)

    ### convert variables from tf checkpoints to caffemodel
    with tf.Session() as sess:
        saver.restore(sess,  checkpoint)
        tf_all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # for i, var in enumerate(tf_all_vars):
        #     print(i, var.name, var.shape.as_list())
        print('------------------------------------------------------------------')
        i = 0   # index
        for caffe_var_name in net.params.keys():
            for n in range(len(net.params[caffe_var_name])):
                if list(net.params[caffe_var_name][n].data.shape) != [1]:
                    var = tf_all_vars[i]
                    print(i, caffe_var_name, net.params[caffe_var_name][n].data.shape, var.name, var.shape.as_list())
                    i += 1
        # exit()
        """ tf name scope:
        convolutional layer:
        "MobilenetV2/....../...weights:0"
        "MobilenetV2/....../BatchNorm/gamma:0"
        "MobilenetV2/....../BatchNorm/beta:0"
        "MobilenetV2/....../BatchNorm/moving_mean:0"
        "MobilenetV2/....../BatchNorm/moving_variance:0"
        fully connected layer:
        "MobilenetV2/....../...weights:0"
        "MobilenetV2/....../biases:0"
        """

        #            name,           shape list
        # caffe_var: caffe_var_name, list(net.params[caffe_var_name][n].data.shape)
        # tf_var   : tf_var.name,    tf_var.shape.as_list()

        ### 262 variables to convert from tf.ckpt to caffemodel

        i = 0   # index
        for caffe_var_name in net.params.keys():
            for n in range(len(net.params[caffe_var_name])):
                if list(net.params[caffe_var_name][n].data.shape) != [1]:

                    ### Compare caffe_var and tf_var here

                    # caffe_var_name = caffe_var_name
                    caffe_var_data = net.params[caffe_var_name][n].data
                    caffe_var_shape = list(caffe_var_data.shape)

                    tf_var_name = tf_all_vars[i].name
                    tf_var_shape = tf_all_vars[i].shape.as_list()
                    if 'weights:0' in tf_var_name:
                        ### weight layer
                        # print(caffe_var_name, caffe_var_shape, '|||||||||||', tf_var_name, tf_var_shape)

                        tf_var_data = sess.run(tf_all_vars[i])

                        ### swap tf_var axis for caffe_var:
                        ### tf_var shape: (height, width, channel_out, channel_in) for depthwise_weights
                        ###               (height, width, channel_in, channel_out) for other weights
                        ### caffe_var shape: (channel_out, channel_in, height, width)

                        tf_var_data = np.transpose(tf_var_data, axes=(3, 2, 0, 1))

                        if '/depthwise_weights' in tf_var_name:
                            tf_var_data = np.swapaxes(tf_var_data, axis1=0, axis2=1)

                        if 'Logits/' in tf_var_name:
                            ### mismatched num_classes
                            ### tf class 0: 'background'
                            caffe_var_data[:, ...] = tf_var_data[1:, ...]
                        else:
                            caffe_var_data[...] = tf_var_data[...]

                    if 'biases:0' in tf_var_name:
                        ### bias layer
                        # print(caffe_var_name, caffe_var_shape, '|||||||||||', tf_var_name, tf_var_shape)
                        ### tf_var_shape: (1001,)
                        ### caffe_var_shape: (1000,)
                        tf_var_data = sess.run(tf_all_vars[i])
                        caffe_var_data[:] = tf_var_data[1:]


                    if 'BatchNorm/gamma:0' in tf_var_name:
                        ### batchnorm scaling layer, but convert mean
                        # print(caffe_var_name, n, caffe_var_shape, '|||||||||||', tf_all_vars[i+2].name, tf_all_vars[i+2].shape.as_list())
                        ### tf_var_shape: (channel,)
                        ### caffe_var_shape: (channel,)
                        tf_var_data = sess.run(tf_all_vars[i+2])
                        caffe_var_data[...] = tf_var_data[...]

                    if 'BatchNorm/beta:0' in tf_var_name:
                        ### batchnorm scaling layer, but convert variance
                        # print(caffe_var_name, n, caffe_var_shape, '|||||||||||', tf_all_vars[i+2].name, tf_all_vars[i+2].shape.as_list())
                        ### tf_var_shape: (channel,)
                        ### caffe_var_shape: (channel,)
                        tf_var_data = sess.run(tf_all_vars[i+2])
                        caffe_var_data[...] = tf_var_data[...] # + 1e-3 -1e-5

                    if 'BatchNorm/moving_mean:0' in tf_var_name:
                        ### batchnorm moving average layer, but convert gamme
                        # print(caffe_var_name, n, caffe_var_shape, '|||||||||||', tf_all_vars[i-2].name, tf_all_vars[i-2].shape.as_list())
                        ### tf_var_shape: (channel,)
                        ### caffe_var_shape: (channel,)
                        tf_var_data = sess.run(tf_all_vars[i-2])
                        caffe_var_data[...] = tf_var_data[...]

                    if 'BatchNorm/moving_variance:0' in tf_var_name:
                        ### batchnorm moving average layer, but convert beta
                        # print(caffe_var_name, n, caffe_var_shape, '|||||||||||', tf_all_vars[i-2].name, tf_all_vars[i-2].shape.as_list())
                        ### tf_var_shape: (channel,)
                        ### caffe_var_shape: (channel,)
                        tf_var_data = sess.run(tf_all_vars[i-2])
                        caffe_var_data[...] = tf_var_data[...]
                    i+=1
                else:
                    ### moving average factor, must set to 1
                    net.params[caffe_var_name][n].data[...] = 1.
                    # print(caffe_var_name, n, list(net.params[caffe_var_name][n].data.shape), '|||||||||||', net.params[caffe_var_name][n].data)

    net.save(to_caffemodel)
    print('Save converted caffemodel to', to_caffemodel)
    return net


def tf_preprocess(image_file):
    tf.reset_default_graph()
    file_input = tf.placeholder(tf.string, ())
    images = tf.image.decode_jpeg(tf.read_file(file_input))

    images = tf.expand_dims(images, 0)
    images = tf.cast(images, tf.float32) / 128. - 1
    images.set_shape((None, None, None, 3))
    images = tf.image.resize_images(images, (image_scale, image_scale))
    with tf.Session() as sess:
        tf_image = sess.run(images, feed_dict={file_input: image_file})

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # (image_scale, image_scale, 3) to (3, image_scale, image_scale)
    image = transformer.preprocess('data', tf_image[0])
    return image


def caffe_test_from_ckpt(net, image_file, preprocess_from_caffe=True):
    ### data preprocessing
    if preprocess_from_caffe:
        im = caffe.io.load_image(image_file)
        ## crop square image in the center
        # h, w, _ = im.shape
        # if h < w:
        #     off = (w - h) / 2
        #     im = im[:, off:off + h]
        # else:
        #     off = (h - w) / 2
        #     im = im[off:off + h, :]

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))  # (image_scale, image_scale, 3) to (3, image_scale, image_scale)
        transformer.set_mean('data', 128./255*np.ones([3])) # scale [-0.5, 0.5]
        transformer.set_input_scale('data', 255./128) # scale [-1, 1]
        image = transformer.preprocess('data', im)
    else: # preprocess from tf
        image = tf_preprocess(image_file)

    net.blobs['data'].data[...] = image[...]
    out = net.forward()

    print('....................................')
    print('file to be predicted:', image_file)
    prob = out['prob']
    prob = np.squeeze(prob)
    idx = np.argsort(-prob)
    label_names = np.loadtxt('synset.txt', str, delimiter='\t')
    for i in range(5):
        label = idx[i]
        print('%.2f - %s' % (prob[label], label_names[label]), label)


def compare_layer_output(net, layer_name, checkpoint, tensor_name, image_file):
    ### Compare outputs from the same layer (tensor)
    ### from caffe net and tensorflow graph

    ### matching name examples:
    ##    tf: MobilenetV2/Conv/Conv2D:0, MobilenetV2/Conv/Relu6:0, MobilenetV2/Conv/BatchNorm/FusedBatchNorm:0
    ## caffe: conv1/sliced2,             conv1/relu,               conv1/scale

    def square_error(x, x_):
        return np.sum(np.square(x-x_))

    image = tf_preprocess(image_file)

    ## caffe inference
    net.blobs['data'].data[...] = image[...]
    net.forward()
    caffe_output = net.blobs[layer_name].data
    caffe_output = caffe_output.transpose(0, 2, 3, 1) # channel first to last

    ## tf inference
    tf.reset_default_graph()
    images = tf.placeholder(tf.float32, shape=(None, image_scale, image_scale, 3))
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
        logits, endpoints = mobilenet_v2.mobilenet(images, num_classes=1001)
    ema = tf.train.ExponentialMovingAverage(0.999)
    vars = ema.variables_to_restore()
    saver = tf.train.Saver(vars)

    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        tensor = sess.graph.get_tensor_by_name(tensor_name)
        tf_output = sess.run(tensor, feed_dict={images: image})

    ### compare tf and caffe result of a specific layer
    ### need graphs and layer (tensor) name in caffe and tf
    print('...................................')
    error = 0
    for i in range(32):
        err = square_error(tf_output[0, :, :, i], caffe_output[0, :, :, i])
        print('channel', i, err)
        error += err
    print('total error:', error)
    print('...................................')

    return


if __name__ == '__main__':
    net = caffe_load_from_ckpt(prototxt, checkpoint, to_caffemodel)
    caffe_test_from_ckpt(net, image_file, preprocess_from_caffe=True)
