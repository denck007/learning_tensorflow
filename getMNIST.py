
# utility functions for downloading and extracting MNIST data
# Code it taken from the tensorflow tutorial 3_mnist_from_scratch


import os
from six.moves.urllib.request import urlretrieve
    
import gzip, binascii, struct, numpy

def downloadMNIST(working_dir = "/tmp/mnist-data"):

    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
    WORK_DIRECTORY = working_dir

    def maybe_download(filename):
        """A helper to download the data files if not present."""
        if not os.path.exists(WORK_DIRECTORY):
            os.mkdir(WORK_DIRECTORY)
        filepath = os.path.join(WORK_DIRECTORY, filename)
        if not os.path.exists(filepath):
            filepath, _ = urlretrieve(SOURCE_URL + filename, filepath)
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        else:
            print('Already downloaded', filename)
        return filepath

    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')
    
    return (train_data_filename, train_labels_filename, test_data_filename, test_labels_filename)
    



def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
  
    For MNIST data, the number of channels is always 1.

    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    IMAGE_SIZE = 28
    PIXEL_DEPTH = 255
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        # Skip the magic number and dimensions; we know these values.
        bytestream.read(16)

        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
        return data

def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    NUM_LABELS = 10
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        # Skip the magic number and count; we know these values.
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    # Convert to dense 1-hot representation.
    return (numpy.arange(NUM_LABELS) == labels[:, None]).astype(numpy.float32)

