import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import csv
from tflearn.data_preprocessing import ImagePreprocessing

# Building 'VGG Network'
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

# Building 'VGG Network'
TRAIN_DIR = 'train/'
DIRS = (
'airport_inside', 'bakery', 'bedroom', 'greenhouse', 'gym', 'kitchen', 'operating_room', 'poolinside', 'restaurant',
'toystore')
IMG_SIZE = 224
MODEL_NAME = 'in_doors'


######## Data Agumention########


def crop_image(img):
    # TensorFlow. 'x' = A placeholder for an image.
    original_size = [img.shape[0], img.shape[1], 3]
    x = tf.placeholder(dtype=tf.float32, shape=original_size)
    # Use the following commands to perform random crops
    crop_size = [int(2 * original_size[0] / 3), int(2 * original_size[1] / 3), 3]
    seed = np.random.randint(1234)
    k = tx = tf.random_crop(x, size=crop_size, seed=seed)

    o = tf.image.resize_images(k, [IMG_SIZE, IMG_SIZE])

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        ii = sess.run(o, {x: img})

        ## cropImg =cv2.resize()
        return ii


def Add_noise(img):
    # TensorFlow. 'x' = A placeholder for an image.
    original_size = [img.shape[0], img.shape[1], 3]
    x = tf.placeholder(dtype=tf.float32, shape=original_size)
    # Use the following commands to perform random crops
    crop_size = [int(2 * original_size[0] / 3), int(2 * original_size[1] / 3), 3]
    seed = np.random.randint(1234)
    ##k = tx = tf.random_crop(x, size = crop_size, seed = seed)
    k = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=1.0,
                         dtype=tf.float32)
    output = tf.add(x, k)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        ii = sess.run(output, {x: img})

        ## cropImg =cv2.resize()
        return ii


def mixed_aug(img):
    img = np.fliplr(img)
    img = crop_image(img)
    img = Add_noise(img)
    return img


######## Read Data And Apply Data Agumention########
def create_train_data():
    training_data = []
    for i in range(len(DIRS)):
        cnt = 0
        j = 0
        for img in tqdm(os.listdir(TRAIN_DIR + DIRS[i])):
            path = os.path.join(TRAIN_DIR + DIRS[i], img)
            img_data = cv2.imread(path, 1)
            if img_data is not None:
                img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
                ## img_data = preprocessing_img(img_data)
                a = [0] * 10
                a[i] = 1
                if i == 3 or i == 4 or i == 6 or i == 7 or i == 9:
                    flipped_img = np.fliplr(img_data)
                    cnt = cnt + 1
                    training_data.append([np.array(flipped_img), a])
                if i == 3 or i == 4 or i == 6 or i == 7:
                    croped_img = crop_image(img_data)
                    cnt = cnt + 1
                    training_data.append([np.array(croped_img), a])
                if i == 3 or i == 6 or i == 7:
                    Noised_img = Add_noise(img_data)
                    cnt = cnt + 1
                    training_data.append([np.array(Noised_img), a])

                if i == 3:
                    mixed_img = mixed_aug(img_data)
                    cnt = cnt + 1
                    training_data.append([np.array(mixed_img), a])
                if (i == 1 and j < 200) or (i == 0 and j < 50) or (i == 8 and j < 115):
                    flipped_img = np.fliplr(img_data)
                    cnt = cnt + 1
                    training_data.append([np.array(flipped_img), a])
                    j = j + 1

                cnt = cnt + 1
                training_data.append([np.array(img_data), a])
            else:
                print(path)
        print(cnt)
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


if (os.path.exists('train_data.npy')):  # If you have already created the dataset:
    train_data = np.load('train_data.npy', allow_pickle=True)

else:  # If dataset is not created:
    train_data = create_train_data()

train = train_data

X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
X_train = X_train / 255

y_train = [i[1] for i in train]

img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center(mean=[123.68, 116.779, 103.939], per_channel=True)

######## Build Vgg archi ########

tf.reset_default_graph()

input_layer = input_data(shape=[None, 224, 224, 3])

block1_conv1 = conv_2d(input_layer, 64, 3, activation='relu', name='block1_conv1')
block1_conv2 = conv_2d(block1_conv1, 64, 3, activation='relu', name='block1_conv2')
block1_pool = max_pool_2d(block1_conv2, 2, strides=2, name='block1_pool')

block2_conv1 = conv_2d(block1_pool, 128, 3, activation='relu', name='block2_conv1')
block2_conv2 = conv_2d(block2_conv1, 128, 3, activation='relu', name='block2_conv2')
block2_pool = max_pool_2d(block2_conv2, 2, strides=2, name='block2_pool')

block3_conv1 = conv_2d(block2_pool, 256, 3, activation='relu', name='block3_conv1')
block3_conv2 = conv_2d(block3_conv1, 256, 3, activation='relu', name='block3_conv2')
block3_conv3 = conv_2d(block3_conv2, 256, 3, activation='relu', name='block3_conv3')
block3_conv4 = conv_2d(block3_conv3, 256, 3, activation='relu', name='block3_conv4')
block3_pool = max_pool_2d(block3_conv4, 2, strides=2, name='block3_pool')

block4_conv1 = conv_2d(block3_pool, 512, 3, activation='relu', name='block4_conv1')
block4_conv2 = conv_2d(block4_conv1, 512, 3, activation='relu', name='block4_conv2')
block4_conv3 = conv_2d(block4_conv2, 512, 3, activation='relu', name='block4_conv3')
block4_conv4 = conv_2d(block4_conv3, 512, 3, activation='relu', name='block4_conv4')
block4_pool = max_pool_2d(block4_conv4, 2, strides=2, name='block4_pool')

block5_conv1 = conv_2d(block4_pool, 512, 3, activation='relu', name='block5_conv1')
block5_conv2 = conv_2d(block5_conv1, 512, 3, activation='relu', name='block5_conv2')
block5_conv3 = conv_2d(block5_conv2, 512, 3, activation='relu', name='block5_conv3')
block5_conv4 = conv_2d(block5_conv3, 512, 3, activation='relu', name='block5_conv4')
block4_pool = max_pool_2d(block5_conv4, 2, strides=2, name='block4_pool')
flatten_layer = tflearn.layers.core.flatten(block4_pool, name='Flatten')

fc1 = fully_connected(flatten_layer, 4096, activation='relu')

# layer below this are not restored!


regression = tflearn.regression(fc1, optimizer='adam',
                                loss='categorical_crossentropy',
                                learning_rate=0.001, restore=False)

model = tflearn.DNN(regression, checkpoint_path='vgg-finetuning',
                    max_checkpoints=3, tensorboard_verbose=2,
                    tensorboard_dir="./logs")

model_path = ''
model_file = os.path.join(model_path, "vgg19.tflearn")
model.load(model_file, weights_only=True)

######## Transfer Learing ########

x = np.empty((X_train.shape[0], 4096))

for i in range(0, X_train.shape[0], 100):
    if i + 100 > X_train.shape[0]:
        gg = X_train.shape[0] % 100
        X_features = model.predict(X_train[i:i + gg])
        x[i:i + gg] = X_features

    X_features = model.predict(X_train[i:i + 100])
    x[i:i + 100] = X_features

tf.reset_default_graph()
input_layer = tflearn.input_data(shape=[None, 4096])

dense1 = tflearn.fully_connected(input_layer, 4096, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(dense1, 0.5)
dense2 = tflearn.fully_connected(dropout1, 512, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(dense2, 0.5)

softmax = tflearn.fully_connected(dropout2, 10, activation='softmax')

net = tflearn.regression(softmax, optimizer='adam', learning_rate=0.0001,
                         loss='categorical_crossentropy')

# Training
model1 = tflearn.DNN(net, tensorboard_verbose=0)
model1.fit(x, y_train, n_epoch=120, validation_set=0,
           show_metric=True, run_id="model")


######## Testing ########

test_dic = 'test/'
with open('Predictions.csv', mode='w') as test_file:
    test_writer = csv.writer(test_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    test_writer.writerow(['id', 'label'])

    for t_i in os.listdir(test_dic):

        path = os.path.join(test_dic, t_i)
        img = cv2.imread(path, 1)
        if img is not None:

            test_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            ## test_img = preprocessing_img(test_img)
            test_img = test_img.reshape(IMG_SIZE, IMG_SIZE, 3)
            test_img = test_img / 255
            prediction = model.predict([test_img])[0]
            pp = model1.predict([prediction])[0]
            p = np.argmax(pp)
            p = p + 1

            test_writer.writerow([t_i, p])
        else:
            print(t_i)



