import tensorflow as tf
import os
from PIL import Image
import numpy as np
import glob

# extracting labels for the eyeglasses column
def extract_labels(file_path):
    with open(file_path, 'r') as f:
        num_image = int(f.readline().strip())
        print("{} records".format(num_image))
        attributes = f.readline().strip().split(" ")
        #print(len(attributes))
        for i in range(len(attributes)):
            if attributes[i] == "Eyeglasses":
                break;
        print(i)
        # got index of eyeglasses
        # load only those values
        labels = np.loadtxt(f, usecols=[i+1], dtype = np.int, skiprows=2)

        labels[np.where(labels<0)] = 0
    print("finished loading labels") 
    return labels

# this method opens the images one by one and loads the images
# we resize the images to the resolution we want to work on (52x52x3 in this case)
def extract_images(file_path):
    #files = [f for f in glob.glob("/Users/manpreetdhanjal/Downloads/img_align_celeba")]
    files = [f for f in glob.glob(file_path)]
    data =np.empty((1,52,52,3))
    j = 0
    for f in files:
        img =  glob.glob(f +"/*.jpg")
        
        for i in img:
            j = j+1;
            arr = get_image_data(i)
            data = np.append(data,arr,axis=0)
            print('data is',(data).shape)
        
    data = data[1:,:]
    return data

# this method returns the image as nd array
def get_image_data(filename):
    sz = 52
    imgData = (Image.open(filename))
    imgData = imgData.resize((sz,sz))
    imgData = np.asarray(imgData)
    imgData = imgData.reshape(1, sz,sz, 3)
    return imgData

# initialising weights with a normal distribution and standard deviation
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01, dtype=tf.float32) 
    return tf.Variable(initial)

# initialise bias variables
def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)

# initialise convolution layer
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# initialise max pool layer
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def train_CNN(celeb_images, celeb_labels, learning_param):

	# divide into training, validation and testing data set
	train_input = celeb_images[0:40000,:]
	train_labels = celeb_labels[0:40000,:]
	validation_input = celeb_images[40000:50000,:]
	validation_labels = celeb_labels[40000:50000,:]
	test_input = celeb_images[50000:60000,:]
	test_labels = celeb_labels[50000:60000,:]

	# input
	# 52x52x3 = 8112 - input layer
	# only one node on the output layer
	x = tf.placeholder(tf.float32, [None, 8112])
	y_conv = tf.placeholder(tf.float32, [None, 1])

	# parameter
	W = tf.Variable(tf.zeros([8112, 1], dtype=tf.float32))
	b = tf.Variable(tf.zeros([1], dtype=tf.float32))

	# logistic regression using sigmoid
	y = tf.nn.sigmoid(tf.matmul(x, W) + b)
	# cross entropy
	y_ = tf.placeholder(tf.float32, [None, 1])

	# initiate weights for layer 1, convolution layer 1 and pool layer 1
	# third parameter is 3 since we have 3D image
	W_conv1 = weight_variable([5, 5, 3, 32])
	b_conv1 = bias_variable([32])
	x_image = tf.reshape(x, [-1, 52, 52, 3])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	# initiate weights for layer 2, convolution layer 2 and pool layer 2
	W_conv2 = weight_variable([5, 5, 32, 64]) 
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	# initiate weights for layer 3 and fully connected layer
	W_fc1 = weight_variable([13 * 13 * 64, 1024]) 
	b_fc1 = bias_variable([1024])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 13*13*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	# specify dropout value
	keep_prob = tf.placeholder(tf.float32) 
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	# weights for output layer
	# second dimension is 1 since we have to classify in 2 classes
	W_fc2 = weight_variable([1024, 1]) 
	b_fc2 = bias_variable([1])
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	# declare the function to be minimised
	cross_entropy = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y_conv))
	# use optimiser for gradient descent
	train_step = tf.train.AdamOptimizer(learning_param).minimize(cross_entropy) 
	# check the number of correct predictions
	correct_prediction = tf.equal(tf.round(y_conv), y_)
	# method to calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# define the size of minibatch
	batch_size = 500
	data_size = 40000

	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    for i in range(int(data_size/batch_size)):
	        start = i
	        end = min(i+batch_size, data_size);
	        # divided labels and input images into minibatch
	        epoch_x = train_input[start:end, :]
	        epoch_y = train_labels[start:end, :]
	      
	        train_accuracy = accuracy.eval(feed_dict={x: epoch_x, y_: epoch_y, keep_prob: 1.0})
	        print('step %d, training accuracy %g' % (i, train_accuracy)) 
	        
	        # train the model
	        train_step.run(feed_dict={x: epoch_x, y_: epoch_y, keep_prob:0.8})
	        # print the test accuracy
	        print('test accuracy %g' % accuracy.eval(feed_dict={x: test_input, y_: test.labels, keep_prob: 1.0}))

	    return accuracy.eval(feed_dict={x: validation_input, y_:validation_labels, keep_prob: 1.0})


def tune_CNN(celeb_images, celeb_labels):
	
	# grid search
    eta_arr = [1e-8, 1e-9, 1e-10]
    accuracy = 0
    min_accuracy = 2
    opt_param = 0;

  	for learning_param in eta_arr:
		accuracy = train_SNN(celeb_images, celeb_labels,learning_param)
		if(accuracy < min_accuracy):
			opt_param = eta_arr

	return eta_arr


def main():

	# extract the images and respective labels for eyeglasses
	celeb_images = extract_images("/Users/manpreetdhanjal/Downloads/img_align_celeba");
	celeb_labels = extract_labels("/Users/manpreetdhanjal/Downloads/list_attr_celeba.txt");

	# taking only 60000 images for training
	celeb_images = celeb_images.reshape(-1, 52*52*3)
	celeb_labels = celeb_labels[0:60000]
	celeb_labels = celeb_labels.reshape(60000, 1);

	eta = tune_CNN(celeb_images, celeb_labels)

	# training the SNN
	train_CNN(celeb_images, celeb_labels, eta)

main()

