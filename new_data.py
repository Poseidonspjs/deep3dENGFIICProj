import os
import io
import sys
import mxnet as mx
import numpy as np
import cv2
import logging
import lmdb
import random
from datetime import datetime
from PIL import Image
from multiprocessing.pool import ThreadPool
import argparse

# -----------------------
# Helper functions
# -----------------------
def get_prop(txn, key, default, check=False):
    v = txn.get(key.encode())
    if v is None:
        txn.put(key.encode(), str(default).encode())
        v = default
    else:
        v = type(default)(v.decode())
    if check:
        assert v == default, f"{v} != {default}"
    return v

def crop_img(img, p, shape, margin=0, test=False, grid=1):
    if p is None:
        if test:
            p = ((img.shape[1]-shape[0]-margin)//grid//2, (img.shape[0]-shape[1])//grid//2)
        else:
            p = (random.randint(0, (img.shape[1]-shape[0]-margin)//grid),
                 random.randint(0, (img.shape[0]-shape[1])//grid))
    return img[p[1]*grid:p[1]*grid+shape[1], p[0]*grid:p[0]*grid+shape[0]], p

def load_mean(fname, data_iter, label_mean=False, include=None):
    if not fname.endswith('.npz'):
        fname += '.npz'
    if os.path.isfile(fname):
        return np.load(fname)
    else:
        print(f"Mean file {fname} not found. Computing mean...")
        data_iter.reset()
        mean_list = [np.zeros(shape[1:], dtype=np.float64) for name, shape in data_iter.provide_data]
        name_list = [name for name, shape in data_iter.provide_data]
        if label_mean:
            mean_list += [np.zeros(shape[1:], dtype=np.float64) for name, shape in data_iter.provide_label]
            name_list += [name for name, shape in data_iter.provide_label]
        count = 0
        last_mean_list = None
        for batch in data_iter:
            arr_list = batch.data
            if label_mean:
                arr_list += batch.label
            for name, arr, mean in zip(name_list, arr_list, mean_list):
                if include is None or name in include:
                    arr = arr.asnumpy()
                    mean += arr[:arr.shape[0]-batch.pad].sum(axis=0)
            inc = batch.data[0].shape[0]-batch.pad
            count += inc
            if count//1000 > (count-inc)//1000:
                print(f"processed {count}")
                if last_mean_list is None:
                    last_mean_list = [np.zeros_like(mean, dtype=np.float64) for mean in mean_list]
                flag = True
                for mean, last_mean in zip(mean_list, last_mean_list):
                    cur_mean = mean/count
                    if not np.isclose(last_mean, cur_mean).all():
                        print(np.max(np.abs(last_mean-cur_mean)))
                        flag = False
                        last_mean[:] = cur_mean
                        break
                mean_dict = dict(zip(name_list, [(mean/count).astype(np.float32) for mean in mean_list]))
                np.savez(fname, **mean_dict)
                if flag:
                    break
        data_iter.reset()
        return mean_dict

# -----------------------
# Data iterator
# -----------------------
class Mov3dStack(mx.io.DataIter):
    def __init__(self, path, data_shape, batch_size, scale,
                 mean_file=None, test_mode=False, output_depth=False, data_frames=1, flow_frames=1,
                 source=None, upsample=1, base_shape=None, stride=1, no_left0=False, right_whiten=False):
        self.data_shape = data_shape
        self.batch_size = batch_size
        self.scale = scale
        self.test_mode = test_mode
        self.output_depth = output_depth
        self.data_frames = data_frames
        self.flow_frames = flow_frames
        self.upsample = upsample
        self.base_shape = base_shape
        self.stride = stride
        self.no_left0 = no_left0
        self.right_whiten = right_whiten
        self.fix_p = None

        self.env = lmdb.open(path, map_size=1<<40, max_dbs=5, readonly=True, readahead=False)
        self.ldb = self.env.open_db(b'l')
        if flow_frames > 0:
            self.fdb = self.env.open_db(b'flow')
        if output_depth:
            self.ddb = self.env.open_db(b'depth')
            self.margin = (scale[1] - scale[0])//2
        else:
            self.margin = 0
        self.rdb = self.env.open_db(b'r')

        self.cur = 0
        with self.env.begin() as txn:
            if source:
                self.idx = [int(i) for i in txn.get(source.encode()).decode().split(',')]
            else:
                self.idx = [int(i) for i in txn.get(b'shuffled_test_idx').decode().split(',')]

        # Setup provide_data / provide_label
        self.provide_data = []
        if data_frames > 0:
            self.provide_data.append(('left', (batch_size, 3*data_frames, data_shape[1], data_shape[0])))
        if flow_frames > 0:
            self.provide_data.append(('flow', (batch_size, 2*flow_frames, data_shape[1], data_shape[0])))
        if not no_left0:
            self.provide_data.append(('left0', (batch_size, 3, data_shape[1]*upsample, data_shape[0]*upsample)))
        self.provide_label = [('l1_label', (batch_size, 3, data_shape[1]*upsample, data_shape[0]*upsample))]
        if self.output_depth:
            self.provide_label = [('softmax_label', (batch_size, data_shape[1]*data_shape[0]))]

        self.left_mean = np.zeros((3, data_shape[1], data_shape[0]))
        self.right_mean = np.zeros((3, data_shape[1], data_shape[0]))
        self.left_mean_nd = mx.nd.array(self.left_mean)
        self.left_mean_nd_1 = self.left_mean_nd.reshape((1,) + self.left_mean_nd.shape)
        self.right_mean_nd = mx.nd.array(self.right_mean)

        if flow_frames > 0:
            self.flow_mean = np.zeros((2, data_shape[1], data_shape[0]))
            self.flow_mean_nd = mx.nd.array(self.flow_mean)

        if mean_file is None:
            mean_file = os.path.join(path, 'mean.npz')
        mean_dict = load_mean(mean_file, self, label_mean=True)
        self.left_mean = mean_dict['left']
        self.right_mean = mean_dict['l1_label']
        if flow_frames > 0:
            self.flow_mean = mean_dict['flow']
            self.flow_mean_nd = mx.nd.array(self.flow_mean)

        self.left_mean_nd = mx.nd.array(self.left_mean)
        self.left_mean_nd_1 = self.left_mean_nd.reshape((1,) + self.left_mean_nd.shape)
        self.right_mean_nd = mx.nd.array(self.right_mean)

    def reset(self):
        logging.info(f"Mov3dStack.reset at {self.cur}")
        self.cur = 0
        if not self.test_mode:
            random.shuffle(self.idx)

    def seek(self, n_iter):
        self.cur = (n_iter*self.batch_size) % len(self.idx)

    def __iter__(self):
        return self

    def __next__(self):
        # For simplicity, stop iteration at end
        if self.cur >= len(self.idx):
            raise StopIteration
        # Implement data loading here (similar to original next() logic)
        # Return mx.io.DataBatch
        raise NotImplementedError("Data loading logic needs to be filled per your dataset")
