import random
import string

import redis

import classes
import settings

users_db = redis.StrictRedis(host=settings.REDIS_HOST,
                             port=settings.REDIS_PORT,
                             decode_responses=True,
                             db=settings.USER_DB)


def get_users():
    """
    Returns a list of all users, wherein each user is a Python dict.
    """
    all_users = []

    for redis_name_key in users_db.scan_iter(match='user:*'):
        user = get_user(name=redis_name_key[5:])
        all_users.append(user)

    return all_users


def get_user(name=None, token=None):
    """
    Returns a user as a Python dict, based on the specified name OR token.
    If both specified, name takes precedence.
    If user is not found, raises UserNotFoundError.
    """
    if name is None and token is None:
        raise classes.UserNotFoundError('Specify a name or token.')
    elif name is not None:
        token = users_db.get('user:' + name)

    if token is None:
        raise classes.UserNotFoundError('No user of name <{}> found.'.format(name))

    user = users_db.hgetall(token)
    if user == {}:
        raise classes.UserNotFoundError('No user of token <{}> found.'.format(token))

    return user


def delete_user(name=None, token=None):
    """
    Deletes a user by name OR token. If both specified, name takes precedence.
    Returns the deleted user as a Python dict.
    If user not found, raises UserNotFoundError.
    """
    if name is None and token is None:
        raise classes.UserNotFoundError('Specify a name or token.')
    elif name is not None:
        token = users_db.get('user:' + name)

    if token is None:
        raise classes.UserNotFoundError('No user of name <{}> found.'.format(name))

    user = users_db.hgetall(token)
    if user == {}:
        raise classes.UserNotFoundError('No user of token <{}> found.'.format(token))

    user = users_db.hgetall(token)
    if user == {}:
        raise classes.UserNotFoundError('No user of token <{}> found.'.format(token))

    users_db.delete('user:' + name)
    users_db.delete(token)

    return user


def add_user(user):
    """
    Adds a new user based on partial user info specified in a Python dict.
    Returns the new user's info in a Python dict.
    If the information provided is invalid, raises UserInfoError.
    If an existing user already has the same name or token, raises UserConflictError.
    """
    complete_user = populate_missing_info(user)
    redis_name_key = 'user:' + complete_user['name']

    token = users_db.get(redis_name_key)
    if token is not None:
        raise classes.UserConflictError('User with the name <{}> already exists.'.format(complete_user['name']))
    if users_db.hgetall(complete_user['token']) != {}:
        raise classes.UserConflictError('User with the token <{}> already exists.'.format(complete_user['token']))

    users_db.set(name=redis_name_key, value=complete_user['token'])
    users_db.hmset(name=complete_user['token'], mapping=user)
    return get_user(name=user['name'])


def update_user(name, updates):
    """
    Updates a user.
    Returns a Python dict containing the new user.
    If existing user is not found, raises UserNotFoundError.
    If the new information provided is invalid, raises UserInfoError.
    If the new information provided conflicts with an existing user, raises UserConflictError.
    """
    old_user = get_user(name=name)

    if updates.get('name') is not None and old_user['name'] != updates['name']:
        try:
            conflicting_user = get_user(name=updates['name'])
        except classes.UserNotFoundError:
            check_name(updates['name'])
        else:
            raise classes.UserConflictError(
                'An existing user with the name <{}> already exists.'.format(updates['name'])
            )
    else:
        updates['name'] = old_user['name']

    if updates.get('token') is not None:
        try:
            conflicting_user = get_user(token=updates['token'])
        except classes.UserNotFoundError:
            check_token(updates['token'])
        else:
            raise classes.UserConflictError(
                'An existing user with the token <{}> already exists.'.format(updates['token'])
            )
    else:
        updates['token'] = old_user['token']

    if updates.get('tier') is not None:
        check_tier(updates['tier'])
        old_user['tier'] = updates['tier']
    else:
        updates['tier'] = old_user['tier']

    if updates.get('quota') is not None:
        check_quota(updates['quota'], updates['tier'])
    else:
        updates['quota'] = old_user['quota']

    delete_user(name=old_user['name'])
    add_user(updates)
    return get_user(name=updates.get('name'))


def populate_missing_info(user):
    if user.get('name') is None:
        raise classes.UserInfoError('The user\'s name must be specified')
    else:
        check_name(name=user['name'])

    if user.get('token') is None:
        user['token'] = generate_token()
    else:
        check_token(token=user['token'])

    if user.get('tier') is None:
        user['tier'] = 'basic'
    else:
        check_tier(tier=user['tier'])

    if user.get('quota') is None:
        user['quota'] = settings.QUOTA_LIMITS[user['tier']]
    else:
        check_quota(quota=user['quota'], tier=user['tier'])

    return user


def check_name(name):
    if not isinstance(name, str):
        raise classes.UserInfoError('The specified name must be a string.')
    elif not name.isalnum():
        raise classes.UserInfoError('The specified name can only consist of lowercase alphabets and numbers.')
    elif name.lower() != name:
        raise classes.UserInfoError('The specified name cannot container uppercase letters.')
    elif len(name) > 128:
        raise classes.UserInfoError('The specified name cannot exceed 128 characters in length.')
    return


def check_token(token):
    if not isinstance(token, str):
        raise classes.UserInfoError('The specified token must be a string.')
    elif len(token) > 64:
        raise classes.UserInfoError('The specified token cannot exceed 64 characters in length.')
    elif not token.isalnum():
        raise classes.UserInfoError('The specified token can only comprise alphabets and numbers.')
    return


def check_tier(tier):
    if tier not in settings.QUOTA_LIMITS:
        valid_tiers = settings.QUOTA_LIMITS.keys()
        raise classes.UserInfoError('The specified tier must be chosen from {}'.format(set(valid_tiers)))
    return


def check_quota(quota, tier):
    if not isinstance(quota, int):
        raise classes.UserInfoError('The specified quota must be an integer.')
    elif quota < 0:
        raise classes.UserInfoError('The specified quota cannot be negative')
    elif quota > settings.QUOTA_LIMITS[tier]:
        raise classes.UserInfoError(
            'The specified quota cannot exceed tier limits. Tier is now {}. '
            'Limits are : {}.'.format(tier, settings.QUOTA_LIMITS)
        )
    return


def generate_token():
    """
    Generates a random 32-character token as a string.
    Tokens may only include lowercase letters, uppercase letters, and digits.
    They are hence case-sensitive.
    """
    candidate_characters = string.ascii_uppercase + string.ascii_lowercase + string.digits
    new_token = ""
    for i in range(0, 32):
        new_token += random.choice(candidate_characters)
    return new_token

print(get_users())
