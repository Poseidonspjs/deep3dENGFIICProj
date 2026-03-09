import mxnet as mx
import numpy as np
from sym import make_l1_sym
import argparse
import logging
import data
import cv2


def make_movie(model_path, ctx, source='test_idx', fname=None):
    from parse import anaglyph, sbs
    from parse import split

    logging.basicConfig(level=logging.DEBUG)

    n_batch = 50
    batch_size = 10
    start = 0
    scale = (-15, 17)

    data_frames = 1
    flow_frames = 0
    upsample = 4

    data_shape = (384, 160)
    udata_shape = (data_shape[0]*upsample, data_shape[1]*upsample)
    base_shape = (432*upsample, 180*upsample)

    method = 'multi2'
    pos = [(24,10), (0,0), (0,20), (48,0), (48,20)]

    norm = np.zeros((3, base_shape[1], base_shape[0]), dtype=np.float32)

    boarder = 32*upsample
    feather = np.ones((udata_shape[1], udata_shape[0]), dtype=np.float32)

    for i in range(udata_shape[1]):
        for j in range(udata_shape[0]):
            feather[i, j] = min((min(i, j)+1.0)/boarder, 1.0)

    for p in pos:
        up = (p[0]*upsample, p[1]*upsample)
        norm[:, up[1]:up[1]+udata_shape[1], up[0]:up[0]+udata_shape[0]] += feather

    metric_xy = mx.metric.MAE()
    metric_py = mx.metric.MAE()

    test_data = data.Mov3dStack(
        'data/lmdb',
        data_shape,
        batch_size,
        scale,
        output_depth=False,
        data_frames=data_frames,
        flow_frames=flow_frames,
        test_mode=True,
        source=source,
        upsample=upsample,
        base_shape=base_shape
    )

    sym = make_l1_sym(scale, 'sum', data_frames, flow_frames, False,
                      method=method, upsample=upsample)

    init = mx.init.Load(model_path, verbose=True)

    model = mx.model.FeedForward(
        ctx=ctx,
        symbol=sym,
        initializer=init
    )

    model._init_params(dict(test_data.provide_data + test_data.provide_label))

    lcap = cv2.VideoWriter()
    pcap_ana = cv2.VideoWriter()
    pcap_sbs = cv2.VideoWriter()
    ycap_ana = cv2.VideoWriter()
    ycap_sbs = cv2.VideoWriter()

    width = base_shape[0]
    height = base_shape[1]

    fourcc = cv2.VideoWriter_fourcc(*'X264')

    lcap.open(fname+'_l.mkv', fourcc, 24, (width, height))
    pcap_ana.open(fname+'_ana_p.mkv', fourcc, 24, (width, height))
    pcap_sbs.open(fname+'_sbs_p.mkv', fourcc, 24, (width, height))
    ycap_ana.open(fname+'_ana_y.mkv', fourcc, 24, (width, height))
    ycap_sbs.open(fname+'_sbs_y.mkv', fourcc, 24, (width, height))

    data_names = [x[0] for x in test_data.provide_data]

    model._init_predictor(test_data.provide_data)

    data_arrays = [model._pred_exec.arg_dict[name] for name in data_names]

    for batch_idx in range(n_batch):

        X = np.zeros((batch_size,)+norm.shape, dtype=np.float32)
        Y = np.zeros((batch_size,)+norm.shape, dtype=np.float32)
        P = np.zeros((batch_size,)+norm.shape, dtype=np.float32)

        for p in pos:

            test_data.seek(start+batch_idx)
            test_data.fix_p = p

            batch = test_data.next()

            mx.executor._load_data(batch, data_arrays)

            model._pred_exec.forward(is_train=False)

            up = (p[0]*upsample, p[1]*upsample)

            X[:, :, up[1]:up[1]+udata_shape[1], up[0]:up[0]+udata_shape[0]] += batch.data[-1].asnumpy() * feather
            Y[:, :, up[1]:up[1]+udata_shape[1], up[0]:up[0]+udata_shape[0]] += batch.label[0].asnumpy() * feather
            P[:, :, up[1]:up[1]+udata_shape[1], up[0]:up[0]+udata_shape[0]] += model._pred_exec.outputs[0].asnumpy() * feather

        X = np.clip(X/norm, 0, 255)
        Y = np.clip(Y/norm, 0, 255)
        P = np.clip(P/norm, 0, 255)

        metric_py.update([mx.nd.array(Y)], [mx.nd.array(P)])
        metric_xy.update([mx.nd.array(Y)], [mx.nd.array(X)])

        print(batch_idx, metric_py.get(), metric_xy.get())

        X = X.astype(np.uint8).transpose((0, 2, 3, 1))
        Y = Y.astype(np.uint8).transpose((0, 2, 3, 1))
        P = P.astype(np.uint8).transpose((0, 2, 3, 1))

        for i in range(X.shape[0]):
            lcap.write(X[i])
            ycap_ana.write(anaglyph(X[i], Y[i]))
            ycap_sbs.write(sbs(X[i], Y[i]))
            pcap_ana.write(anaglyph(X[i], P[i]))
            pcap_sbs.write(sbs(X[i], P[i]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert movie to 3D')

    parser.add_argument('model_path', type=str)
    parser.add_argument('--source', type=str, default='test_idx')
    parser.add_argument('--output', type=str, default='output')

    args = parser.parse_args()

    make_movie(args.model_path, mx.cpu(), args.source, args.output)
