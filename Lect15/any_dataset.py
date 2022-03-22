import threading
from sklearn.utils import shuffle
import numpy as np



class threadsafe_iter:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def get_objects_i(objects_count):
    """Cyclic generator of paths indices
    """
    current_objects_id = 0
    while True:
        yield current_objects_id
        current_objects_id = (current_objects_id + 1) % objects_count


class MyDataset():
    """Sun Landmarks dataset."""

    def __init__(self, fname, batch_size = 32, shuffle = True):
        self.data = self.read_my_data(fname)
        self.shuffle = shuffle
        self.shuffle_data()
        self.batch_size = batch_size

        self.objects_id_generator = threadsafe_iter(get_objects_i(len(self.data)))

        self.lock = threading.Lock()  # mutex for input path
        self.yield_lock = threading.Lock()  # mutex for generator yielding of batch
        self.init_count = 0
        self.mask_dict = {}

    def __len__(self):
        return len(self.data)


    def shuffle_data(self):
        if self.shuffle:
            self.data = shuffle(self.data)

    def read_my_data(self, fname):
        # here we need to read the data from disk
        return np.random.randn(100, 21)
        # raise NotImplementedError()

    def get_data_by_id(self, id):
        # here we need to get our data by the ID: index, filename, etc.
        return self.data[id]
        # raise NotImplementedError()

    def read_mask(self, mask_fname):
        if mask_fname in self.mask_dict.keys():
            return self.mask_dict[mask_fname]
        else:
            return self.read_mask_by_fname(mask_fname)


    def __iter__(self):
        while True:
            with self.lock:
                if (self.init_count == 0):
                    self.shuffle_data()
                    self.batch_data = []
                    self.init_count = 1

            for obj_id in self.objects_id_generator:
                curr_data_x, curr_data_mask, curr_data_y = self.get_data_by_id(obj_id)

                # augmentation

                curr_data_x = np.expand_dims(curr_data_x, 0)
                curr_data_mask = ...
                curr_data_y = ...

                # here we may apply some augmentations

                # Concurrent access by multiple threads to the lists below
                with self.yield_lock:
                    if (len(self.batch_data)) < self.batch_size:
                        self.batch_data.append((curr_data_x, curr_data_mask, curr_data_y))

                    if len(self.batch_data) % self.batch_size == 0:
                        # self.batch_data = np.concatenate(self.batch_data, axis=0)
                        batch_x = np.concatenate(([a[0] for a in self.batch_data]), axis=0)
                        batch_mask = np.concatenate(([a[1] for a in self.batch_data]), axis=0)
                        batch_y = np.concatenate(([a[2] for a in self.batch_data]), axis=0)

                        yield batch_x, batch_mask, batch_y
                        # yield self.batch_data
                        self.batch_data = []