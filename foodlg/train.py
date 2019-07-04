# ./spearmint ../examples/food-product-vgg16/config.pb --driver=local --method=GPEIOptChooser --method-args=noiseless=0

'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
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

#from nn_transfer import transfer
import numpy as np
import argparse
import os
import random as rn
import json

from .se_resnet import SEResNet101
from .se_inception_resnet_v2 import SEInceptionResNetV2
from .dpn import DPN92
from .shufflenet import ShuffleNet
from .resnet152 import ResNet152
from .resnext import ResNextImageNet as ResNext
from .tool import preprocess_input

MAX_EPOCH = 100


class LearningRateReducer(Callback):

    def __init__(self, patience=0, reduce_rate=0.5, reduce_nb=10, verbose=1):
        super(Callback, self).__init__()
        self.patience = patience
        self.wait = 0
        self.best_score = 10000
        self.reduce_rate = reduce_rate
        self.current_reduce_nb = 0
        self.reduce_nb = reduce_nb
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        print('lr: %f, wait: %d' %
              (K.get_value(self.model.optimizer.lr), self.wait))
        current_score = logs.get('val_loss')
        if current_score < self.best_score:
            self.best_score = current_score
            self.wait = 0
            if self.verbose > 0:
                print('---current best val accuracy: %.3f' % current_score)
        else:
            if self.wait >= self.patience:
                self.current_reduce_nb += 1
                if self.current_reduce_nb <= self.reduce_nb:
                    lr = float(K.get_value(self.model.optimizer.lr))
                    K.set_value(self.model.optimizer.lr, lr * self.reduce_rate)
                    self.wait = 0
                else:
                    if self.verbose > 0:
                        print("Epoch %d: early stopping" % (epoch))
                    self.model.stop_training = True
            else:
                self.wait += 1

# currently not used


def get_lr(epoch, lr):
    if epoch >= MAX_EPOCH / 4 and epoch < MAX_EPOCH * 2 / 4:
        lr = 0.1 * lr
    elif epoch >= MAX_EPOCH * 2 / 4 and epoch < MAX_EPOCH * 3 / 4:
        lr = 0.01 * lr
    elif epoch >= MAX_EPOCH * 3 / 4:
        lr = 0.001 * lr
    return lr


def train(job_id, args):
    batch_size = args.batchsize
    lr = args.lr
    momentum = args.momentum
    # dimensions of our images.
    img_width, img_height = args.imgwidth, args.imgheight

    train_data_dir = os.path.join(args.basedir, args.dataset, 'train')
    validation_data_dir = os.path.join(
        args.basedir, args.dataset, 'validation')
    nb_train_samples = args.numoftrainings
    nb_validation_samples = args.numofvalidations
    epochs = MAX_EPOCH

    if args.dataset == 'food101':
        classes = 101
    elif args.dataset == 'food172':
        classes = 172
    elif args.dataset == 'food191':
        classes = 191
    elif args.dataset == 'uec100':
        classes = 100
    elif args.dataset == 'uec256':
        classes = 256
    else:  # aggregated dataset
        classes = 668

    if args.model == 'xception':  # 299, 299, 3
        base_model = Xception(
            include_top=True, weights='imagenet', input_shape=(img_height, img_width, 3))
    elif args.model == 'inceptionresnet':  # 299, 299, 3
        base_model = InceptionResNetV2(
            include_top=True, weights='imagenet', input_shape=(img_height, img_width, 3))
    elif args.model == 'nasnetlarge':  # 331, 331, 3
        base_model = NASNetLarge(
            include_top=True, weights='imagenet', input_shape=(img_height, img_width, 3))
    elif args.model == 'nasnetmobile':  # 224, 224, 3
        base_model = NASNetMobile(
            include_top=True, weights='imagenet', input_shape=(img_height, img_width, 3))
    elif args.model == 'densenet169':  # 224, 224, 3
        base_model = DenseNet169(
            weights='imagenet', input_shape=(img_height, img_width, 3))
    elif args.model == 'se-resnet':
        base_model = SEResNet101(weights='imagenet')
    elif args.model == 'se-inceptionresnet':
        base_model = SEInceptionResNetV2(weights='imagenet')
    elif args.model == 'shufflenet':
        base_model = ShuffleNet(input_shape=(
            img_height, img_width, 3), groups=args.group, bottleneck_ratio=args.bn_ratio)
    elif args.model == 'mobilenet':  # 224, 224, 3
        base_model = MobileNet(input_shape=(
            img_height, img_width, 3), weights='imagenet', alpha=args.alpha)
    elif args.model == 'mobilenetv2':  # 224, 224, 3
        base_model = MobileNetV2(input_shape=(
            img_height, img_width, 3), weights='imagenet', alpha=args.alpha)
    elif args.model == 'dpn':
        base_model = DPN92((img_height, img_width, 3), weights='imagenet')
    elif args.model == 'resnet50':
        base_model = ResNet50(
            include_top=True, weights='imagenet', input_shape=(img_height, img_width, 3))
    elif args.model == 'resnet152':
        base_model = ResNet152(include_top=True, weights=(args.pretrain_weights if args.pretrain_weights else None),
                               input_shape=(img_height, img_width, 3))
    elif args.model == 'resnext':
        base_model = ResNext(
            include_top=True, input_shape=(img_height, img_width, 3))
    else:
        base_model = None

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
    model = Model(input=base_model.input, output=[predictions])

    # fine-tune model with previous weights
    if args.finetune_weights != '':
        model.load_weights(args.finetune_weights)
    model.summary()

    if float(args.weight_decay) != 0:
        for layer in model.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = regularizers.l2(args.weight_decay)
            if hasattr(layer, 'bias_regularizer'):
                layer.bias_regularizer = regularizers.l2(args.weight_decay)

        for layer in model.layers:
            if hasattr(layer, 'kernel_regularizer'):
                print(layer.name, layer.kernel_regularizer.l2)

    sgd = optimizers.SGD(lr=lr, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    # store model architecture
    # model_json = model.to_json()
    # with open("vgg19_nolastpooling_model_256.json", "w") as json_file:
    #     json_file.write(model_json)

    # this is the augmentation configuration we will use for training
    if 'dpn' in args.model or 'resne' in args.model:  # or 'shufflenet' in args.model:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)
    else:
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    # store class indices
    print train_generator.class_indices
    np.save('class_indices/' + args.dataset +
            '.npy', train_generator.class_indices)

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    # print K.get_value(model.optimizer.lr)
    if args.checkpoint == '':
        prefix = args.basedir + '/' + args.dataset + '/models/' + args.model
    else:
        prefix = args.checkpoint + '/' + args.model

    if 'mobilenet' in args.model:
        prefix += '-' + str(args.alpha)
    callbacks = [
        #EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        ModelCheckpoint(prefix + '-{epoch:02d}-{val_acc:.2f}.h5', monitor='val_loss',
                        save_best_only=True, save_weights_only=True, verbose=1),
        CSVLogger(prefix + '_log.csv', append=True, separator=','),
        # LearningRateScheduler(get_lr),
        LearningRateReducer(patience=5, reduce_rate=0.1, reduce_nb=3),
    ]

    if args.initialepoch == -1:
        train_history = model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size, callbacks=callbacks)
    else:  # resume training
        train_history = model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size, callbacks=callbacks,
            initial_epoch=args.initialepoch)

    loss = train_history.history['loss']

    return min(loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train models for food classification.')
    parser.add_argument('--model', '-m', choices=['xception', 'inceptionresnet', 'mobilenet', 'nasnetlarge',
                                                  'nasnetmobile', 'densenet169', 'se-resnet', 'se-inceptionresnet', 'dpn', 'shufflenet',
                                                  'resnet50', 'resnet152', 'resnext', 'mobilenetv2'],
                        default='xception', help='Choose a training model, default is xception net.')
    parser.add_argument('--dataset', '-d', choices=['food101', 'food172', 'food191', 'uec100', 'uec256', 'foodlg'],
                        default='food101', help='Choose a training dataset, default is food101 dataset.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for model training.')
    parser.add_argument('--batchsize', '-b', type=int,
                        default=32, help='Batch size for training at a time.')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--imgwidth', type=int, default=256,
                        help='input image width to the model.')
    parser.add_argument('--imgheight', type=int, default=256,
                        help='input image height to the model.')
    parser.add_argument('--basedir', default='',
                        help='base directory where the datasets are stored.')
    parser.add_argument('--numoftrainings', type=int,
                        default=60600, help='the number of training images.')
    parser.add_argument('--numofvalidations', type=int,
                        default=15150, help='the number of validation images.')
    parser.add_argument('--pretrain_weights', '-pw', default='',
                        help='the path to the pretrained model weights.')
    parser.add_argument('--finetune_weights', '-fw', default='',
                        help='the path to the finetune model weights.')
    parser.add_argument('--checkpoint', default='',
                        help='directory where the checkpoint models will be stored during training, default path will be "models" directory \
		inside the dataset folder, such as basedir/foodlg/models/')
    parser.add_argument('--deviceid', type=int, default=-1,
                        help='device gpu id to train the model, if set to -1, all available gpu resources will be taken up.')
    parser.add_argument('--initialepoch', '-ini', type=int, default=-1,
                        help='initial epoch to train the model, only useful for recovering/fine-tune models from intermedia checkpoints.')
    parser.add_argument('--group', type=int, default=3,
                        help='groups for group convolution operators, only used for certain models.')
    parser.add_argument('--bn_ratio', type=float, default=1.0,
                        help='batch normalization ratio.')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='scale factor for mobilenet.')
    parser.add_argument('--weight_decay', '-wd', type=float,
                        default=1e-4, help='weight dacay during training.')
    parser.add_argument('--load_config', type=str, default='',
                        help='path to the training configuration file, if used, parameter settings above will depend on the configuration file.')
    args = parser.parse_args()

    # load configurations from json file
    if args.load_config != '':
        with open(args.load_config) as f:
            data = json.load(f)
        if 'model' in data:
            args.model = data['model']
        if 'dataset' in data:
            args.dataset = data['dataset']
        if 'lr' in data:
            args.lr = data['lr']
        if 'batchsize' in data:
            args.batchsize = data['batchsize']
        if 'momentum' in data:
            args.momentum = data['momentum']
        if 'imgwidth' in data:
            args.imgwidth = data['imgwidth']
        if 'imgheight' in data:
            args.imgheight = data['imgheight']
        if 'basedir' in data:
            args.basedir = data['basedir']
        if 'numoftrainings' in data:
            args.numoftrainings = data['numoftrainings']
        if 'numofvalidations' in data:
            args.numofvalidations = data['numofvalidations']
        if 'pretrain_weigths' in data:
            args.pretrain_weights = data['pretrain_weights']
        if 'finetune_weights' in data:
            args.finetun_weights = data['finetune_weights']
        if 'checkpoint' in data:
            args.checkpoint = data['checkpoint']
        if 'deviceid' in data:
            args.deviceid = data['deviceid']
        if 'initialepoch' in data:
            args.initialepoch = data['initialepoch']
        if 'group' in data:
            args.group = data['group']
        if 'bn_ratio' in data:
            args.bn_ratio = data['bn_ratio']
        if 'alpha' in data:
            args.alpha = data['alpha']
        if 'weight_decay' in data:
            args.weight_decay = data['weight_decay']
        print args.imgwidth, args.imgheight

    if int(args.deviceid) != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.deviceid)
    os.environ['PYTHONHASHSEED'] = '0'

    # given a fixed initial random state
    np.random.seed(32)
    rn.seed(123)
    tf.set_random_seed(1234)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.visible_device_list='0'
    session = tf.Session(config=config)
    K.set_session(session)

    train(rn.randint(0, 1000), args)
