import numpy as np
import sys
sys.path.append('models/research/slim')
import tensorflow as tf
from nets.mobilenet import mobilenet_v2

base_name = 'mobilenet_v2_1.0_224'
checkpoint = 'mobilenet_v2_1.0_224/' + base_name + '.ckpt'


tf.reset_default_graph()
# For simplicity we just decode jpeg inside tensorflow.
# But one can provide any input obviously.
file_input = tf.placeholder(tf.string, ())

image = tf.image.decode_jpeg(tf.read_file(file_input))

images = tf.expand_dims(image, 0)
images = tf.cast(images, tf.float32) / 128. - 1
images.set_shape((None, None, None, 3))
images = tf.image.resize_images(images, (224, 224))

# images = tf.placeholder(tf.float32, (None, 224, 224, 3))

# Note: arg_scope is optional for inference.
with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
    logits, endpoints = mobilenet_v2.mobilenet(images)

# Restore using exponential moving average since it produces (1.5-2%) higher
# accuracy
ema = tf.train.ExponentialMovingAverage(0.999)
vars = ema.variables_to_restore()

saver = tf.train.Saver(vars)


from datasets import imagenet
with tf.Session() as sess:
    saver.restore(sess,  checkpoint)
    x = endpoints['Predictions'].eval(feed_dict={file_input: 'imgs/dog.jpg'})

    # writer = tf.summary.FileWriter("TensorBoard/", graph=sess.graph)
    # writer.close()


label_map = imagenet.create_readable_names_for_imagenet_labels()
print("Top 1 prediction: ", x.argmax(), label_map[x.argmax()], x.max())