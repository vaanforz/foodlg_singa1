from keras import backend as K

def preprocess_input(x, data_format=None):
    """Preprocesses a tensor encoding a batch of images.
       Obtained from https://github.com/cypw/DPNs

        # Arguments
            x: input Numpy tensor, 3D.
            data_format: data format of the image tensor.

        # Returns
            Preprocessed tensor.
        """
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        x = x[::-1, :, :]
        # Zero-center by mean pixel
        x[0, :, :] -= 104
        x[1, :, :] -= 117
        x[2, :, :] -= 124
    else:
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 104
        x[:, :, 1] -= 117
        x[:, :, 2] -= 124

    x *= 0.0167
    return x

