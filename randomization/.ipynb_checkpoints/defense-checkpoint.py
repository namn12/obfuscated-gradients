import tensorflow as tf

# implements defense described in https://openreview.net/forum?id=Sk9yuql0Z
#
# except they don't mention what value they pad with, so we assume it's 0.5
PAD_VALUE = 0.5 #scalar pad value to use for the padded points

# input_tensor should be of shape [1, 299, 299, 3]
# output is of shape [1, 331, 331, 3]
def defend(input_tensor):
    rnd = tf.random_uniform((), 299, 331, dtype=tf.int32)
    rescaled = tf.image.crop_and_resize(input_tensor, [[0, 0, 1, 1]], [0], [rnd, rnd]) #takes in image, bounding box locations, box index, and desired output crop size; extracts crop from input tensor and resizes them
    h_rem = 331 - rnd
    w_rem = 331 - rnd
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32) #random values from uniform distribution from 0 to w_rem
    pad_right = w_rem - pad_left
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=PAD_VALUE) #pad left and pad right: those indices indicate how much padding to add before and after the image in the 1st and 2nd dimensions
    padded.set_shape((1, 331, 331, 3))
    return padded
