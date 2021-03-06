import torch
import h5py


class Dataset:
    def __init__(self, hdf5_path, patch_h, patch_w, n_channels, batch_size, data_type, thresholds=(), labels=True, empty=False):

        self.i = 0
        self.batch_size = batch_size
        self.done = False
        self.thresholds = thresholds
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.n_channels = n_channels
        self.data_type = data_type

        self.labels_flag = labels
        self.hdf5_path = hdf5_path
        if not empty:
            self.images, self.labels = self.get_hdf5_data()

            self.size = len(self.images)
            self.iterations = len(self.images)//self.batch_size + 1
        else:
            self.images = list()
            self.labels = list()
            self.size = len(self.images)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch(self.batch_size)

    @property
    def shape(self):
        return [len(self.images), self.patch_h, self.patch_w, self.n_channels]

    def get_hdf5_data(self):
        hdf5_file = h5py.File(self.hdf5_path, 'r')
        images = hdf5_file['%s_img' % self.data_type]
        if self.labels_flag:
            labels = hdf5_file['%s_labels' % self.data_type]
        else:
            labels = list()
        return images, labels

    def set_pos(self, i):
        self.i = i

    def get_pos(self):
        return self.i

    def reset(self):
        self.set_pos(0)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_thresholds(self, thresholds):
        self.thresholds = thresholds

    def adapt_label(self, label):
        thresholds = self.thresholds + (None,)
        adapted = [0.0 for _ in range(len(thresholds))]
        i = None
        for i, threshold in enumerate(thresholds):
            if threshold is None or label < threshold:
                break
        adapted[i] = label if len(adapted) == 1 else 1.0
        return adapted

    def next_batch(self, n: int):
        if self.done:
            self.done = False
            raise StopIteration
        batch_img = torch.from_numpy(self.images[self.i:self.i + n])
        batch_labels = torch.from_numpy(self.labels[self.i:self.i + n])
        self.i += len(batch_img)
        delta = n - len(batch_img)
        if delta == n:
            raise StopIteration
        if 0 < delta:
            batch_img = torch.cat((batch_img, torch.from_numpy(self.images[:delta])), axis=0)
            batch_labels = torch.cat((batch_labels, torch.from_numpy(self.labels[:delta])), axis=0)
            self.i = delta
            self.done = True
        batch_img = batch_img.permute(0,3,1,2)
        return batch_img/255.0, batch_labels
