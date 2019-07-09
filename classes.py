import pickle
import datetime

import settings


class Request:
    def __init__(self, image, task=None):
        self.id = datetime.datetime.today().strftime(settings.DATETIME_FORMAT)
        self.image = image
        if(task is None):
            self.task = 'food204'
        else:
            self.task = task
        self.results = None

    def __repr__(self):
        return '[Request] ' + self.id

    def __str__(self):
        return '[Request] ' + self.id

    def as_serialized(self):
        return pickle.dumps(self)

    @staticmethod
    def from_serialized(serial_bytes):
        return pickle.loads(serial_bytes, encoding='bytes')


class UserAuthenticationError(TypeError):
    def __init__(self, message=None):
        super().__init__(message)


class ImageNotFoundError(TypeError):
    def __init__(self, message=None):
        super().__init__(message)


class InvalidTaskError(ValueError):
    def __init__(self, message=None):
        super().__init__(message)


class AppServerTimeoutError(ValueError):
    def __init__(self, message=None):
        super().__init__(message)


class JSONNotFoundError(TypeError):
    def __init__(self, message=None):
        super().__init__(message)


class UserNotFoundError(LookupError):
    def __init__(self, message=None):
        super().__init__(message)


class UserConflictError(ValueError):
    def __init__(self, message=None):
        super().__init__(message)


class UserInfoError(ValueError):
    def __init__(self, message=None):
        super().__init__(message)
