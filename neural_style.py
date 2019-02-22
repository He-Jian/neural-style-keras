import numpy as np
import tensorflow
import keras
import keras.backend as K
from keras.optimizers import Adam
from keras.applications import vgg16
from keras.layers import Input
import os
from PIL import Image
from skimage.transform import resize
from scipy.misc import imsave

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
size = 512

content_layers = ['block4_conv2']
style_layers = ['block1_conv1', 'block2_conv1',
                'block3_conv1', 'block4_conv1',
                'block5_conv1']


def load_image(image_file):
    image = Image.open(image_file)
    image_array = np.asarray(image.convert("RGB"))
    image_array = image_array / 255.
    image_array = resize(image_array, (size, size))
    image_array = (image_array - imagenet_mean) / imagenet_std
    return image_array


def gram_matrix(x):
    print K.ndim(x), x.shape
    features = K.batch_flatten(K.permute_dimensions(x, (0, 3, 1, 2)))
    gram = K.dot(features, K.transpose(features))
    return gram


def get_style_features(layer_dict, style_layers):
    style_features = [gram_matrix(layer_dict[layer_name]) for layer_name in style_layers]
    return style_features


def content_loss(content, combination):
    '''
    Content loss is simply the MSE between activations of a layer
    '''
    return K.mean(K.stack([K.square(cont - comb) for cont, comb in zip(content, combination)]))


def style_loss(style, combination):
    loss_list = K.stack([K.square(fea - s) / (36.0 * (size ** 2)) for fea, s in zip(combination, style)])
    return K.mean(loss_list)


def tv_loss(x):
    assert K.ndim(x) == 4
    a = K.square(x[:, :-1, :-1, :] - x[:, 1:, :-1, :])
    b = K.square(x[:, :-1, :-1, :] - x[:, :-1, 1:, :])
    return K.mean(a + b, axis=(0, 1, 2, 3))


content_image = np.expand_dims(load_image('content1.jpg'), 0)
style_image = np.expand_dims(load_image('style1.jpg'), 0)
model = vgg16.VGG16(weights='imagenet', include_top=False)
layer_dict = dict([(layer.name, layer.output) for layer in model.layers])
content_features = [layer_dict[layer_name] for layer_name in content_layers]
style_features = get_style_features(layer_dict, style_layers)
get_content_fun = K.function([model.input], content_features)
get_style_fun = K.function([model.input], style_features)
content_targets = get_content_fun([content_image])
style_target = get_style_fun([style_image])
print len(content_targets)
print len(style_target)
print(style_target[0].shape)

target = K.variable(content_image)  # init target image with content image
model2 = vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=Input(tensor=target))
target_layer_dict = dict([(layer.name, layer.output) for layer in model2.layers])
target_image_content_features = [target_layer_dict[layer_name] for layer_name in content_layers]
print target_image_content_features
target_image_style_features = get_style_features(target_layer_dict, style_layers)
cont_loss = content_loss(content_targets, target_image_content_features)
sty_loss = style_loss(style_target, target_image_style_features)
var_loss = tv_loss(target)
# total_loss = K.variable(0.)
total_loss = cont_loss * 5 + sty_loss * 1e-1 + var_loss * 10
from keras.optimizers import Adam

opt = Adam(lr=0.5)
updates = opt.get_updates([target], {}, total_loss)
# List of outputs
outputs = [total_loss]

# Function that makes a step after backpropping to the image
make_step = K.function([], outputs, updates)
import time

# Perform optimization steps and save the results
start_time = time.time()
num_iterations = 10
for i in range(num_iterations):
    out = make_step([])
    print K.get_value(cont_loss) * 5, K.get_value(sty_loss) * 1e-1, K.get_value(var_loss) * 10
    if i % 1 == 0:
        x = K.get_value(target)
        print x.shape
        x = x[0]
        x = 255.0 * (x * imagenet_std + imagenet_mean)
        print x
        img = np.clip(x, 0, 255).astype('uint8')

        fname = str(i) + '.png'
        imsave(fname, img)
