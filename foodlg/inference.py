import os
import inspect

import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Reshape
from keras.layers import Dense, GlobalAveragePooling2D
from keras import layers
from keras import backend as K
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.densenet import DenseNet169
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.applications.resnet50 import ResNet50
from keras import optimizers
from keras import regularizers
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.callbacks import Callback
from keras import backend as K

import numpy as np
import argparse
import os
import random as rn
import json
from PIL import Image as pil_image

from .se_resnet import SEResNet101
from .se_inception_resnet_v2 import SEInceptionResNetV2
from .dpn import DPN92
from .shufflenet import ShuffleNet
from .resnet152 import ResNet152
from .resnext import ResNextImageNet as ResNext
from .tool import preprocess_input


class ConvNet(object):

    def __init__(self, args):
        self.args = args
        if int(args.deviceid) != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.deviceid)
        os.environ['PYTHONHASHSEED'] = '0'

        # given a fixed initial random state
        np.random.seed(32)
        rn.seed(123)
        tf.set_random_seed(1234)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        K.set_session(session)
        # dimensions of the image

        classes = int(''.join([x for x in args.dataset if x.isdigit()]))

        img_size = 224

        if args.model == 'xception':  # 299, 299, 3
            img_size = 299
            base_model = Xception(
                include_top=True, weights=None, input_shape=(img_size, img_size, 3))
        elif args.model == 'inceptionresnet':  # 299, 299, 3
            img_size = 299
            base_model = InceptionResNetV2(
                include_top=True, input_shape=(img_size, img_size, 3))
        elif args.model == 'nasnetlarge':  # 331, 331, 3
            img_size = 331
            base_model = NASNetLarge(
                include_top=True, input_shape=(img_size, img_size, 3))
        elif args.model == 'nasnetmobile':  # 224, 224, 3
            base_model = NASNetMobile(
                include_top=True, input_shape=(img_size, img_size, 3))
        elif args.model == 'densenet169':  # 224, 224, 3
            base_model = DenseNet169(input_shape=(img_size, img_size, 3))
        elif args.model == 'se-resnet':
            base_model = SEResNet101()
        elif args.model == 'se-inceptionresnet':
            base_model = SEInceptionResNetV2()
        elif args.model == 'shufflenet':
            base_model = ShuffleNet(input_shape=(
                img_size, img_size, 3), groups=args.group, bottleneck_ratio=args.bn_ratio)
        elif args.model == 'mobilenet':  # 224, 224, 3
            base_model = MobileNet(input_shape=(
                img_size, img_size, 3),  alpha=args.alpha)
        elif args.model == 'mobilenetv2':  # 224, 224, 3
            base_model = MobileNetV2(input_shape=(
                img_size, img_size, 3),  alpha=args.alpha)
        elif args.model == 'dpn':
            base_model = DPN92((img_size, img_size, 3))
        elif args.model == 'resnet50':
            base_model = ResNet50(
                include_top=True, input_shape=(img_size, img_size, 3))
        elif args.model == 'resnet152':
            base_model = ResNet152(
                include_top=True, input_shape=(img_size, img_size, 3))
        elif args.model == 'resnext':
            base_model = ResNext(
                include_top=True, input_shape=(img_size, img_size, 3))
        else:
            print('model not implemented.')
            return
        args.img_size = img_size
        base_model.layers.pop()
        if args.model == 'mobilenet':
            base_model.layers.pop()
            base_model.layers.pop()
            x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(
                base_model.layers[-1].output)
            x = Activation('softmax', name='act_softmax')(x)
            predictions = Reshape((classes,), name='reshape_2')(x)
        elif args.model == 'shufflenet':
            base_model.layers.pop()
            predictions = Dense(classes, activation='softmax')(
                base_model.layers[-1].output)
        else:
            predictions = Dense(classes, activation='softmax')(
                base_model.layers[-1].output)
        self.model = Model(input=base_model.input, output=[predictions])

        # test model with previous weights
        if args.weights != '':
            self.model.load_weights(args.weights)
        self.model.summary()

        self.graph = tf.get_default_graph()

        label_index_path = os.path.join(os.path.dirname(os.path.abspath(
            inspect.stack()[0][1])), 'class_indices', args.dataset + '.npy')
        label2index = np.load(label_index_path).item()  # dict object
        self.index2label = [None] * len(label2index)
        for k, v in label2index.items():
            self.index2label[v] = k

    def predict(self, img):
        # img: pil img
        with self.graph.as_default():
            width_height_tuple = (self.args.img_size, self.args.img_size)
            # print(img.size)
            img = img.convert('RGB')
            if img.size != width_height_tuple:
                img = img.resize(width_height_tuple, pil_image.NEAREST)
            x = img_to_array(img)
            print(x.shape)
            if 'dpn' in self.args.model or 'resne' in self.args.model:
                x = preprocess_input(x).reshape((1,) + x.shape)
            else:
                x /= 255 * 1.
                x = x.reshape((1,) + x.shape)
            y = self.model.predict(x).reshape((-1,))
            # return topk class names as a list
            top_indexes = np.argsort(y)[::-1][:self.args.topk]
            tops = {}
            for idx in top_indexes:
                tops[self.index2label[idx]] = float(y[idx])
            return tops


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model prediction')
    parser.add_argument('--model', '-m', choices=['xception', 'inceptionresnet', 'mobilenet', 'nasnetlarge',
                                                  'nasnetmobile', 'densenet169', 'se-resnet', 'se-inceptionresnet', 'dpn', 'dpn-ft', 'shufflenet',
                                                  'resnet50', 'resnet152', 'resnext', 'mobilenetv2'],
                        default='xception', help='Choose a training model, default is xception net.')
    parser.add_argument('--dataset', '-d', choices=['food101', 'food172', 'food191', 'uec100', 'uec256', 'foodlg', 'food204'],
                        default='food191', help='Choose a dataset, default is food191 dataset.')
    parser.add_argument('--img_size', type=int, default=224,
                        help='input image width/height. nasnetlarge uses 331; xception uses 299')
    parser.add_argument('--weights', '-w', default='',
                        help='the path to the pretrained model weights')
    parser.add_argument('--device', type=int, default=0,
                        help='device gpu id to put the model.')
    parser.add_argument('--img_path', type=str,
                        help='path to the inference image.')
    parser.add_argument('--topk', type=int, default=5,
                        help='number of labels to return.')
    parser.add_argument('--group', type=int, default=3,
                        help='groups for group convolution operators, only used for certain models.')
    parser.add_argument('--bn_ratio', type=float, default=1.0,
                        help='batch normalization ratio.')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='scale factor for mobilenet.')
    parser.add_argument('--load_config', type=str, default='',
                        help='path to the inference configuration file which overwrites the cmd args')
    args = parser.parse_args()

    # load configurations from json file
    if args.load_config != '':
        with open(args.load_config) as f:
            data = json.load(f)
            if 'model' in data:
                args.model = data['model']
            if 'dataset' in data:
                args.dataset = data['dataset']
            if 'img_size' in data:
                args.img_size = data['img_size']
            if 'weights' in data:
                args.weights = data['weights']
            if 'device' in data:
                args.device = data['device']
            if 'group' in data:
                args.group = data['group']
            if 'bn_ratio' in data:
                args.bn_ratio = data['bn_ratio']
            if 'alpha' in data:
                args.alpha = data['alpha']
            if 'img_path' in data:
                args.image_path = data['img_path']
            if 'topk' in data:
                args.topk = data['topk']

    model = ConvNet(args)
    img = load_img(args.img_path)
    tops = model.predict(img, model, label2index, args)
    print('top-', args.topk, ' classes: ', tops)
