"""
Implement your model here in this module. This models.py file has no dependencies on other project files.
"""
from foodlg import inference
from PIL import Image
import io
import argparse
import requests
import numpy as np
import ast

import os
import inspect
import time
import traceback

import redis

from classes import Request
import settings

from keras.preprocessing.image import load_img, img_to_array

requests_db = redis.StrictRedis(host=settings.REDIS_HOST,
                                port=settings.REDIS_PORT,
                                db=settings.REQUEST_DB)
results_db = redis.StrictRedis(host=settings.REDIS_HOST,
                               port=settings.REDIS_PORT,
                               decode_responses=True,
                               db=settings.RESULT_DB)
server_model = None
server_task = None


class Model(object):

    def __init__(self, args):
        pass

    def predict(self, img):
        pass


class EchoModel(Model):

    def predict(self, img):
        return img


class DummyModel(Model):

    def predict(self, img):
        return {
            'cat': 0.95,
            'dog': 0.82,
            'fish': 0.35,
            'laksa': 0.25
        }


class FoodClassifier(Model):

    def __init__(self, args):
        self.args = args
        self.model = inference.ConvNet(args)

    def predict(self, img):
        img = Image.open(io.BytesIO(img))
        return self.model.predict(img)


class ForwardQueryToRafiki(Model):

    def __init__(self):
        label_index_path = os.path.join(os.path.dirname(os.path.abspath(inspect.stack()[0][1])), 'foodlg', 'class_indices', 'food204' + '.npy')
        label2index = np.load(label_index_path).item()  
        self.index2label = [None] * len(label2index)
        for k, v in label2index.items():
            self.index2label[v] = k

    def predict(self, img):
        img = Image.open(io.BytesIO(img))
        img = img.convert('RGB')
        x = img_to_array(img)

        predictor_host='http://ncrs.d2.comp.nus.edu.sg:44745/predict'
        data={'query': x.tolist()}
        headers = {'Content-Type': 'application/json'}

        r = requests.post(predictor_host, headers=headers, json=data)
        original_pred_output = np.asarray(ast.literal_eval(r.content.decode('utf-8'))['prediction'], dtype=np.float32)
        top_indexes = np.argsort(original_pred_output)[::-1][:5]

        tops = {}
        for idx in top_indexes:
            tops[self.index2label[idx]] = float(original_pred_output[idx])
        
        return tops


def initialize_with_args(args):
    print(' * Loading model...')

    global server_task
    dataset = args.dataset.lower()
    server_task = dataset

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.deviceid)
    os.environ['PYTHONHASHSEED'] = '0'

    global server_model
    if dataset == 'echo':
        server_model = EchoModel(args)
    elif dataset == 'all':
        server_model = DummyModel(args)
    elif dataset == 'rafiki':
        server_model = ForwardQueryToRafiki()
    elif dataset.startswith('food'):
        server_model = FoodClassifier(args)
    else:
        raise ValueError("Unkown server task %s" % (server_task))

    print(' * Successfully loaded model {}.'.format(server_model))
    return


def poll_queue_forever():
    print(' * Polling task queue \'{}\'...'.format(server_task))
    while True:
        try:
            batch = get_one_batch()
            # import pdb; pdb.set_trace()
            process_one_batch(batch)
            sleep()
        except redis.exceptions.ConnectionError:
            print('{}: Could not connect to the Redis server for the requests queue.'
                  .format(time.asctime(time.localtime(time.time()))))
        except Exception as exception:
            print('Error during batch classification')
            traceback.print_exc()
    return


def get_one_batch():
    redis_pipeline = requests_db.pipeline()
    redis_pipeline.multi()
    redis_pipeline.lrange(server_task, 0, settings.BATCH_SIZE_LIMIT - 1)
    redis_pipeline.ltrim(server_task, settings.BATCH_SIZE_LIMIT, -1)
    return redis_pipeline.execute()[0]


def process_one_batch(batch):
    for serialized_request in batch:
        request = Request.from_serialized(serialized_request)
        request.results = process_one_request(request)
        requests_db.set(name=request.id, value=request.as_serialized())


def process_one_request(request):
    print(' * Processing request <{}>... '.format(request.id))
    # try:
    #     if request.task == 'food204_rafiki':
    #         server_model = ForwardQueryToRafiki(args)
    #     else:
    #         pass
    # except:
    #     pass

    server_model = ForwardQueryToRafiki()
    results = server_model.predict(img=request.image)

    if server_task != 'echo':
        save_image_and_result(request_id=request.id,
                              img=request.image, results=results)
    # except Exception as e:  # TODO catch exceptions from model, set exception to a string for the user to see
        # results = str(e)
    print('     Successfully processed request <{}> '.format(request.id))
    return results


def save_image_and_result(request_id, img, results):
    if not os.path.exists(settings.RESULTS_IMG_FOLDER):
        os.makedirs(settings.RESULTS_IMG_FOLDER)

    save_image(request_id=request_id, img=img)
    save_result(request_id=request_id, results=results)
    return


def save_image(request_id, img):
    img_file = open(settings.RESULTS_IMG_FOLDER +
                    '/' + request_id + '.jpg', "wb")
    img_file.write(img)
    img_file.close()
    return


def save_result(request_id, results):
    results_db.hmset(name=request_id, mapping=results)
    return


def sleep():
    time.sleep(settings.MODEL_POLLING_INTERVAL)


def get_args():
    parser = argparse.ArgumentParser(description='model prediction')

    parser.add_argument('--model', '-m', choices=['xception', 'inceptionresnet', 'mobilenet', 'nasnetlarge',
                                                  'nasnetmobile', 'densenet169', 'se-resnet', 'se-inceptionresnet',
                                                  'dpn', 'dpn-ft', 'shufflenet',
                                                  'resnet50', 'resnet152', 'resnext', 'mobilenetv2'],
                        default='xception', help='Choose a training model, default is xception net.')
    parser.add_argument('--dataset', '-d', choices=['food101', 'food172', 'food191', 'uec100', 'uec256', 'foodlg', 'food158',
        'food167', 'food203', 'food204'],
                        default='food101', help='Choose a training dataset, default is food101 dataset.')
    parser.add_argument('--imgwidth', type=int, default=299,
                        help='input image width to the model.')
    parser.add_argument('--imgheight', type=int, default=299,
                        help='input image height to the model.')
    parser.add_argument('--weights', '-w', default='',
                        help='the path to the pretrained model weights')
    parser.add_argument('--deviceid', type=int, default=-1,
                        help='device gpu id to train the model, if set to -1, all available gpu resources will be taken up.')
    parser.add_argument('--topk', type=int, default=5,
                        help='return list, default will be top-1 result.')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    initialize_with_args(args)
    poll_queue_forever()
