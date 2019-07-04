# List of all legal tasks.
#   Keep these one-word long, and in lowercase, since they are also used as command line arguments.
#   If you're adding new tasks:
#     (1) Add to the list below,
#     (2) Map them to your deep learning model, inside the model_server.py module's initialize_with_args() method.
ALL_TASKS = {'echo', 'all', 'food', 'singaporefood'}


# Constants used for requests queuing. Intervals and timeouts are in seconds.
BATCH_SIZE_LIMIT = 1
MODEL_POLLING_INTERVAL = .1
APP_POLLING_INTERVAL = .1
# If no results found within this time frame, then show user a timeout
# error message
APP_POLLING_TIMEOUT = 30


# Redis connection settings.
# If you are modifying this, then also update the redis config files in
# the /databases folder.
REDIS_HOST = 'localhost'
REDIS_PORT = 6379

USER_DB = 0
REQUEST_DB = 1
RESULT_DB = 2

DEFAULT_TASK = 'food204'

# User settings
QUOTA_LIMITS = {
    'basic': 200,
    'paid': 10000,
    'premium': 1000000
}


# Save results to CSV in
DATETIME_FORMAT = '{}-{}-{}-{}-{}-{}-{}'.format(
    '%Y', '%m', '%d', '%H', '%M', '%S', '%f')
RESULTS_IMG_FOLDER = '/root/database/img'
