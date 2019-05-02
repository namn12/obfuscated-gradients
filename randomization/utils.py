import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import PIL.Image
from imagenet_labels import label_to_name
import matplotlib.pyplot as plt

def one_hot(index, total): #makes one hot prediction categories
    '''make hot indices for labels'''
    arr = np.zeros((total))
    arr[index] = 1.0
    return arr

def optimistic_restore(session, save_file):
    '''save the variable checkpoints from all the imagenet labels that have already been run through'''
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = tf.get_variable(saved_var_name)
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

def load_image(path):
    '''load the original image before the shaping, in the proper input image shape'''
    return (np.asarray(PIL.Image.open(path).resize((299, 299)))/255.0).astype(np.float32)

def make_classify(sess, input_, probs):
    '''make two figures, one that creates the cat image, and the other that classifies the classification results'''
    def classify(img, correct_class=None, target_class=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
        fig.sca(ax1)
        p = sess.run(probs, feed_dict={input_: img})[0] #sess is the session that was inputted to function. the .run serves to run the operations and evaluate the tensors in probs)
        ax1.imshow(img)
        fig.sca(ax1)
        
        #obtain the last element of the list (from the last tenth element to the last element)
        topk = list(p.argsort()[-10:][::-1]) #this lets you skip elements (::)
        topprobs = p[topk]
        barlist = ax2.bar(range(10), topprobs) #makes bar plot
        #target class is the class we implement in our attack
        if target_class in topk:
            barlist[topk.index(target_class)].set_color('r')
        if correct_class in topk: #cat
            barlist[topk.index(correct_class)].set_color('g')
        plt.sca(ax2)
        plt.ylim([0, 1.1])
        plt.xticks(range(10),
                   [label_to_name(i)[:15] for i in topk],
                   rotation='vertical')
        fig.subplots_adjust(bottom=0.2)
        plt.show()
    return classify