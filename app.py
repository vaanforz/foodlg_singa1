import json
import time

from flask import Flask, jsonify, redirect, request, render_template, make_response, url_for
from flask_httpauth import HTTPBasicAuth
from passlib.apps import custom_app_context as password_context
import redis

import admins
import classes
import settings
import users

from PIL import Image
import io
import argparse
import requests
import numpy as np
import ast

import os
import inspect
import traceback

from keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
auth = HTTPBasicAuth()
# requests_db = redis.StrictRedis(host=settings.REDIS_HOST,
#                                 port=settings.REDIS_PORT,
#                                 db=settings.REQUEST_DB)


@app.route('/', methods=['GET'])
def index_endpoint():
    return success_response_with_json()


@app.route('/signup', methods=['GET', 'POST'])
def user_signup_endpoint():
    message = None
    if request.method == 'POST':
        try:
            partial_new_user_info = {
                'name': request.form['username'],
            }
            user = users.add_user(user=partial_new_user_info)
        except classes.UserInfoError:
            message = 'Your username must consist of lowercase alphabets and numbers.'
        except classes.UserConflictError:
            message = 'That username has already been taken.'
        else:
            message = 'Your account has been created. Tier: {}. Token: {}'.format(user['tier'], user['token'])
    return render_template('signup.html', error=message)


@app.route('/quota', methods=['GET'])
def user_quota_endpoint():
    try:
        token = get_token(request)
        user = users.get_user(token=token)
    except classes.UserAuthenticationError as bad_token_error:
        return error_response_unauthorized(message=str(bad_token_error))
    except classes.UserNotFoundError:
        return error_response_unauthorized(message='No user associated with the token <{}>.'.format(token))

    return success_response_with_json(quota=user['quota'], tier=user['tier'])


# @app.route('/echo', methods=['POST'])
# def echo_endpoint():
#     try:
#         #print(type(request.data))
#         #print(request.data)
#         image_bytes = construct_image(raw_bytes=request.data)
#         token = get_token(request)
#         users.get_user(token=token)
#     except classes.UserAuthenticationError as bad_token_error:
#         return error_response_unauthorized(message=str(bad_token_error))
#     except classes.UserNotFoundError:
#         return error_response_unauthorized(message='No user associated with your token.')
#     except classes.ImageNotFoundError:
#         return error_response_bad_request(message='Please include an image in your request body.')

#     try:
#         request_id = make_request_in_queue(image=image_bytes, queue='echo')
#         request_done = wait_for_request_done(request_id)
#     except classes.AppServerTimeoutError:
#         return error_response_service_unavailable(message='Request timeout. Please try again.')
#     except Exception as e:
#         print(e)
#         return error_response_internal_server_error(message='Error during echo.')

#     return request_done.results


@app.route('/model', methods=['POST'])
def model_endpoint():
    try:
        image_bytes = construct_image(request.data)  # Raises exception if no bytes in request body.
        token = get_token(request)
        user = users.get_user(token=token)
    except classes.UserAuthenticationError as bad_token_error:
        return error_response_unauthorized(message=str(bad_token_error))
    except classes.UserNotFoundError:
        return error_response_unauthorized(message='No user associated with your token.')
    except classes.ImageNotFoundError:
        return error_response_bad_request(message='Please include an image in your request body.')

    if int(user['quota']) <= 0:
        return error_response_unauthorized(message='No more request quota.', quota=user['quota'], tier=user['tier'])

    def predict_and_parse(image_bytes):
        img = (Image.open(io.BytesIO(image_bytes))).convert('RGB')
        x = img_to_array(img)

        predictor_host='http://ncrs.d2.comp.nus.edu.sg:44745/predict'
        data={'query': x.tolist()}
        headers = {'Content-Type': 'application/json'}

        r = requests.post(predictor_host, headers=headers, json=data)
        original_pred_output = np.asarray(ast.literal_eval(r.content.decode('utf-8'))['prediction'], dtype=np.float32)
        top_indexes = np.argsort(original_pred_output)[::-1][:5]

        label_index_path = os.path.join(os.path.dirname(os.path.abspath(inspect.stack()[0][1])), 'foodlg', 'class_indices', 'food204' + '.npy')
        label2index = np.load(label_index_path).item()  
        index2label = [None] * len(label2index)
        for k, v in label2index.items():
            index2label[v] = k

        tops = {}
        for idx in top_indexes:
            tops[index2label[idx]] = float(original_pred_output[idx])
        
        return tops

    try:
        task = get_task_in_lowercase(request)
        # request_id = make_request_in_queue(image=image_bytes, queue=task)
        # request_done = wait_for_request_done(request_id)
        request_done = predict_and_parse(image_bytes)
        if not task == 'echo':
            newuser = {'quota': int(user['quota']) -1 }
            user = users.update_user(name=user['name'], updates=newuser)
    except classes.InvalidTaskError as invalid_task_error:
        return error_response_bad_request(message=str(invalid_task_error))
    except classes.AppServerTimeoutError:
        return error_response_service_unavailable(message='Request timeout. Please try again.')
    except Exception as e:
        print(e)
        return error_response_internal_server_error(message='Error during classification.')

    return success_response_with_json(quota=user['quota'], tier=user['tier'], results=request_done)


@app.route('/users', methods=['GET'])
@auth.login_required
def all_users_endpoint():
    all_users = users.get_users()
    return success_response_with_json(users=all_users)


@app.route('/users/<string:name>', methods=['GET'])
@auth.login_required
def single_user_endpoint(name):
    try:
        user = users.get_user(name=name)
    except classes.UserNotFoundError as e:
        return error_response_not_found(message=str(e))

    return success_response_with_json(user=user)


@app.route('/users', methods=['POST'])
@auth.login_required
def add_user_endpoint():
    try:
        partial_new_user_info = construct_user(raw_bytes=request.data)
        complete_new_user_info = users.add_user(user=partial_new_user_info)
    except classes.JSONNotFoundError:
        return error_response_bad_request(message='Specify the new user\'s details in a proper JSON request body.')
    except classes.UserInfoError as user_info_error:
        return error_response_bad_request(message=str(user_info_error))
    except classes.UserConflictError as user_conflict_error:
        return error_response_conflict(message=str(user_conflict_error))

    return success_response_with_json(status_code=201, user=complete_new_user_info)


@app.route('/users/<string:name>', methods=['PUT'])
@auth.login_required
def update_user_endpoint(name):
    try:
        old_user = users.get_user(name=name)
        updates = construct_user(request.data)
        updated_user = users.update_user(name=name, updates=updates)
    except classes.UserNotFoundError:
        return error_response_not_found(message='No user associated with that name.')
    except classes.JSONNotFoundError:
        return error_response_bad_request(message='Updated user details are required in a proper JSON request body.')
    except classes.UserInfoError as user_info_error:
        return error_response_bad_request(message=str(user_info_error))
    except classes.UserConflictError as user_conflict_error:
        return error_response_conflict(message=str(user_conflict_error))

    return success_response_with_json(user=updated_user, old_user=old_user)


@app.route('/users/<string:name>', methods=['DELETE'])
@auth.login_required
def delete_user_endpoint(name):
    try:
        deleted_user = users.delete_user(name=name)
    except classes.UserNotFoundError as e:
        return error_response_not_found(message=str(e))

    return success_response_with_json(old_user=deleted_user)


def get_token(request):
    token_detected = (request.args.get('token') or
                      request.headers.get('Authorization') or
                      request.headers.get('Token'))
    if token_detected is None:
        raise classes.UserAuthenticationError('No token detected in your request.')
    elif not isinstance(token_detected, str):
        raise classes.UserAuthenticationError('Your token must be a string.')
    return token_detected


def construct_image(raw_bytes):
    if raw_bytes == b'' or raw_bytes is None:
        raise classes.ImageNotFoundError('Please include an image in your request body.')

    return raw_bytes


def construct_user(raw_bytes):
    try:
        info_dict = json.loads(raw_bytes)
    except json.JSONDecodeError:
        raise classes.JSONNotFoundError(message='Data could not be decoded into a JSON object.')

    if not isinstance(info_dict, dict):
        raise classes.JSONNotFoundError(message='Decoded JSON is not a Python dictionary.')

    return {
        'name': info_dict.get('name'),
        'token': info_dict.get('token'),
        'tier': info_dict.get('tier'),
        'quota': info_dict.get('quota')
    }


def get_task_in_lowercase(request):
    raw_task = request.args.get('task')
    if raw_task is None:
        return settings.DEFAULT_TASK

    result = raw_task.lower()

    # if result not in settings.ALL_TASKS:
    #     raise classes.InvalidTaskError('The specified task <{}> is invalid. '
    #                                    'Valid tasks: {}'.format(result, settings.ALL_TASKS))
    return result


# def make_request_in_queue(image, queue):
#     new_request = classes.Request(image=image, task=queue)
#     #new_request = classes.Request(image=image)
#     requests_db.rpush(queue, new_request.as_serialized())
#     return new_request.id


# def wait_for_request_done(request_id):
#     start_time = time.time()
#     serialized_request_done = None

#     while True:
#         serialized_request_done = requests_db.get(request_id)
#         #print('get results')
#         #print(serialized_request_done)
#         if serialized_request_done is not None:
#             request_done = classes.Request.from_serialized(serialized_request_done)
#             requests_db.delete(request_id)
#             return request_done
#         elif time.time() - start_time > settings.APP_POLLING_TIMEOUT:
#             requests_db.delete(request_id)
#             #print('timeout')
#             raise classes.AppServerTimeoutError()
#         else:
#             time.sleep(settings.APP_POLLING_INTERVAL)


@auth.verify_password
def is_admin(username, password):
    try:
        all_admins = admins.password_hashes
        password_hash = all_admins[username]
    except:  # Admin username not found.
        return False
    return password_context.verify(password, password_hash)  # Admin username found. Verify the password.


@auth.error_handler
def error_response_not_admin():
    return error_response_unauthorized(message='Admin access required.')


@app.errorhandler(400)
def error_response_bad_request(error=None, **kwargs):
    kwargs['error'] = 'Bad Request.'
    return error_response_with_json(error_code=400, **kwargs)


@app.errorhandler(401)
def error_response_unauthorized(error=None, **kwargs):
    kwargs['error'] = 'Unauthorized.'
    return error_response_with_json(error_code=401, **kwargs)


@app.errorhandler(404)
def error_response_not_found(error=None, **kwargs):
    kwargs['error'] = 'Not Found.'
    return error_response_with_json(error_code=404, **kwargs)


@app.errorhandler(405)
def error_response_method_not_allowed(error=None, **kwargs):
    kwargs['error'] = 'Method Not Allowed.'
    return error_response_with_json(error_code=405, **kwargs)


@app.errorhandler(409)
def error_response_conflict(error=None, **kwargs):
    kwargs['error'] = 'Conflict.'
    return error_response_with_json(error_code=409, **kwargs)


@app.errorhandler(500)
def error_response_internal_server_error(error=None, **kwargs):
    kwargs['error'] = 'Internal Server Error.'
    return error_response_with_json(error_code=500, **kwargs)


@app.errorhandler(503)
def error_response_service_unavailable(error=None, **kwargs):
    kwargs['error'] = 'Service Unavailable.'
    return error_response_with_json(error_code=503, **kwargs)


def success_response_with_json(status_code=200, **kwargs):
    kwargs['success'] = True
    return make_response(jsonify(kwargs), status_code)


def error_response_with_json(error_code, **kwargs):
    kwargs['success'] = False
    return make_response(jsonify(kwargs), error_code)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
