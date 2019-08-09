from passlib.apps import custom_app_context as pwd_context

password_hashes = {
    'admin': '$6$rounds=656000$tfiKl7FPQ5VU5YlE$POAeVZC92E/BTExiIDF1l6lPZcyxj6vTKKBsmOiIK78QoW5WKBzkGGGTMlGcs3dEBYvKBwsovHIhCt2Nr1MWc0',
    'rs': '$6$rounds=656000$RqY.pFU8pZE8AK/U$.7mWdlDIw4z9DZfLIqDB.4xP0KnkEO8i2PkHFyEqFVRSizPZ6iZx1E9iaDy4Tb/4WdRuKDjWZVgYHJsWjQWIE0',
    'ww': '$6$rounds=656000$6kgbBYbKIsK7c5YR$rbILuLSxKclU3JhNc8ywUElbKSF1X6bIDVuq4K966VHp/TZrK7SjhkqkI/orccENi5mXwcnS7tZpXexntMG5M.',
    'frontend': '$6$rounds=656000$VdOMAC5zhDEiN5Tc$1mzwlZkCcgSN95p2HRtbew8uhNv7dK/WgzgTIkj7.SmkxowrvW4UhpNsg4vh95d5XzLaRssdWKx4cvmWyQS0n.',
    'vaan': '$6$rounds=656000$Js3r9FWU3YdQlqDq$o2JVb.bppxDeA/fTfnTqAultIk43DAhUIv59kXs1DXPqDYSNN96Kale6ccUiK9ZiAV7TQl5tY8svWRWN9oe.r/'
}


def generate_hash(password):
    return pwd_context.encrypt(password)


def make_new_hash():
    new_password = input('Enter a password to be hashed: ')
    password_hash = generate_hash(new_password)

    results_message = (
        '\nAdd this hash...\n'
        '\n\t{}\n'
        '\n... to the dictionary in the get_password_hashes() function, in the admins.py module.'
    ).format(password_hash)

    print(results_message)


if __name__ == '__main__':
    make_new_hash()

