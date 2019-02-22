import numpy as np
import tensorflow
import keras
import keras.backend as K
from keras.optimizers import Adam
from keras.applications import vgg16
from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Activation, Add
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.engine.topology import Layer
import tensorflow as tf
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

class InstanceNorm(Layer):
    #initialize the layer, and set an extra parameter axis. No need to include inputs parameter!
    def __init__(self,axis=-1, **kwargs):
        self.axis = axis  #-1 for channel last,1 for channel first
        self.result = None
        super(InstanceNorm, self).__init__(**kwargs)
    # first use build function to define parameters, Creates the layer weights.
    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        self.beta = K.variable(K.zeros((1,1,1,input_shape[self.axis])), name='{}_beta'.format(self.name))
        self.gamma = K.variable(K.ones((1,1,1,input_shape[self.axis])), name='{}_gamma'.format(self.name))
        self.trainable_weights = [self.beta, self.gamma ]
        super(InstanceNorm, self).build(input_shape)
    # This is where the layer's logic lives.
    def call(self, x, **kwargs):
        mean, var = tf.nn.moments(x, [1,2], keep_dims=True)
        self.result = tf.nn.batch_normalization(x, mean,var,self.beta,self.gamma,1e-3)
        return self.result
    # return output shape
    def compute_output_shape(self, input_shape):
        #shape = list(input_shape)
        #return tuple([shape[0],shape[-1]])
        return K.int_shape(self.result)

def residual_block(input_tensor, filters=128, kernel_size=3):
    x = Conv2D(filters, (kernel_size, kernel_size),
               padding='same',
               kernel_initializer='he_normal',
               )(input_tensor)
    #x = InstanceNorm(axis=-1)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (kernel_size, kernel_size),
               padding='same',
               kernel_initializer='he_normal',
               )(x)
    #x = InstanceNorm(axis=-1)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def get_transformer(img_input):
    x = Conv2D(32, (9, 9),
               activation='relu',
               padding='same',
               name='trans_conv1')(img_input)
    x = Conv2D(64, (3, 3), strides=2,
               activation='relu',
               padding='same',
               name='trans_conv2')(x)
    x = Conv2D(128, (3, 3), strides=2,
               activation='relu',
               padding='same',
               name='trans_conv3')(x)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='trans_conv4')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(32, (3, 3),
               activation='relu',
               padding='same',
               name='trans_conv5')(x)
    x = Conv2D(3, (3, 3),
               activation='relu',
               padding='same',
               name='trans_conv6')(x)
    model = Model(inputs=img_input, outputs=x)
    return model


def load_image(image_file):
    image = Image.open(image_file)
    image_array = np.asarray(image.convert("RGB"))
    image_array = image_array / 255.
    image_array = resize(image_array, (size, size))
    image_array = (image_array - imagenet_mean) / imagenet_std
    return image_array


def gram_matrix(x):
    # print K.ndim(x), x.shape
    features = K.batch_flatten(K.permute_dimensions(x, (0, 3, 1, 2)))
    gram = K.dot(features, K.transpose(features))
    return gram


def get_style_features(layer_dict, style_layers):
    style_features = [gram_matrix(layer_dict[layer_name]) for layer_name in style_layers]
    return style_features


def content_loss(y_true, y_pred):
    '''
    Content loss is simply the MSE between activations of a layer
    '''
    return K.sum(K.square(y_true - y_pred))


def style_loss1(y_true, y_pred, denom=1.0):
    return K.square(gram_matrix(y_true) - gram_matrix(y_pred)) / (36.0 * (size ** 2))

def style_loss2(y_true, y_pred, denom=1.0):
        return K.square(gram_matrix(y_true) - gram_matrix(y_pred)) / (36.0 * (size ** 2))

def style_loss(y_true, y_pred, denom=1.0):
        return K.square(gram_matrix(y_true) - gram_matrix(y_pred)) / (36.0 * (size ** 2))

def tv_loss(y_true, y_pred):
    x = y_pred
    assert K.ndim(x) == 4
    a = K.square(x[:, :-1, :-1, :] - x[:, 1:, :-1, :])
    b = K.square(x[:, :-1, :-1, :] - x[:, :-1, 1:, :])
    return K.mean(a + b, axis=(0, 1, 2, 3))


vgg16 = vgg16.VGG16(weights='imagenet', include_top=False)
layer_dict = dict([(layer.name, layer.output) for layer in vgg16.layers])
for layer in vgg16.layers:
    layer.trainable = False

'''
out = []
for ln in style_layers:
    out.append(layer_dict[ln])
for ln in content_layers:
    out.append(layer_dict[ln])
'''
out = [layer_dict['block1_conv1'],layer_dict['block2_conv1'],layer_dict['block3_conv1'],layer_dict['block4_conv1'],layer_dict['block5_conv1'],layer_dict['block4_conv2'],]
vgg = Model(inputs=vgg16.input, outputs=out, name='vgg16')
img_input = Input(shape=(size, size, 3))
transformer = get_transformer(img_input)
outs = vgg(transformer.output)
model = Model(inputs=img_input, outputs=[transformer.output, ] + outs)
opt = Adam(lr=0.0001)
model.compile(optimizer=opt,
              loss=[tv_loss, style_loss1, style_loss2, style_loss, style_loss, style_loss, content_loss],
              loss_weights=[20, 1e-11, 1e-11, 1e-11, 1e-11, 1e-11, 1e-6])
plot_model(model, to_file='model.png')
#print(model.summary())

content_image = np.expand_dims(load_image('content1.jpg'), 0)
style_image = np.expand_dims(load_image('style1.jpg'), 0)

original_content_feature = vgg.predict(content_image)[-1]
print original_content_feature.shape
original_style_features = vgg.predict(style_image)
print original_style_features[0].shape
print(len(original_style_features))

checkpoint = ModelCheckpoint('./model.hdf5', monitor='val_loss', verbose=0, save_best_only=False,
                             save_weights_only=False, mode='auto', period=1)

model.fit(content_image, [content_image, ] + original_style_features[:5] + [original_content_feature, ], batch_size=1,
          epochs=2, callbacks=[checkpoint], verbose=1)

img_input = Input(shape=(size, size, 3))
transformer = get_transformer(img_input)
print ("load model weights_path: {}".format('./model.hdf5'))
transformer.load_weights('./model.hdf5', by_name=True)
#print(transformer.summary())
x=transformer.predict(content_image)[0]
x = 255.0 * (x * imagenet_std + imagenet_mean)
print x
img = np.clip(x, 0, 255).astype('uint8')
fname = 'tt.png'
imsave(fname, img)


