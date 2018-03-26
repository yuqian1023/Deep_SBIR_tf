import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from scipy.io import loadmat, savemat
import scipy.spatial.distance as ssd
import glob
from ops import spatial_softmax
import h5py
from skimage.transform import resize
import os


NUM_VIEWS = 10
CROPSIZE = 225


def attentionNet(inputs, pool_method):
    assert (pool_method in ['sigmoid', 'softmax'])
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        trainable=True):
        net = slim.conv2d(inputs, 256, [1, 1], padding='SAME', scope='conv1')
        if pool_method == 'sigmoid':
            logits = slim.conv2d(net, 1, [1, 1], activation_fn=None, scope='conv2')
            prob = tf.nn.sigmoid(logits)
            return prob, logits
        else:
            net = slim.conv2d(net, 1, [1, 1], activation_fn=None, scope='conv2')
            net = spatial_softmax(net)
            return net


def sketch_a_net_sbir(inputs):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        trainable=False):
        with slim.arg_scope([slim.conv2d], padding='VALID'):
            conv1 = slim.conv2d(inputs, 64, [15, 15], 3, scope='conv1_s1')
            conv1 = slim.max_pool2d(conv1, [3, 3], scope='pool1')
            conv2 = slim.conv2d(conv1, 128, [5, 5], scope='conv2_s1')
            conv2 = slim.max_pool2d(conv2, [3, 3], scope='pool2')
            conv3 = slim.conv2d(conv2, 256, [3, 3], padding='SAME', scope='conv3_s1')
            conv4 = slim.conv2d(conv3, 256, [3, 3], padding='SAME', scope='conv4_s1')
            conv5 = slim.conv2d(conv4, 256, [3, 3], padding='SAME', scope='conv5_s1')
            conv5 = slim.max_pool2d(conv5, [3, 3], scope='pool3')
            conv5 = slim.flatten(conv5)
            fc6 = slim.fully_connected(conv5, 512, scope='fc6_s1')
            fc7 = slim.fully_connected(fc6, 256, activation_fn=None, scope='fc7_sketch')
            fc7 = tf.nn.l2_normalize(fc7, dim=1)
    return fc7


def sketch_a_net_dssa(inputs, pool_method='softmax'):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        trainable=False):
        with slim.arg_scope([slim.conv2d], padding='VALID'):
            conv1 = slim.conv2d(inputs, 64, [15, 15], 3, scope='conv1_s1')
            conv1 = slim.max_pool2d(conv1, [3, 3], scope='pool1')
            conv2 = slim.conv2d(conv1, 128, [5, 5], scope='conv2_s1')
            conv2 = slim.max_pool2d(conv2, [3, 3], scope='pool2')
            conv3 = slim.conv2d(conv2, 256, [3, 3], padding='SAME', scope='conv3_s1')
            conv4 = slim.conv2d(conv3, 256, [3, 3], padding='SAME', scope='conv4_s1')
            conv5 = slim.conv2d(conv4, 256, [3, 3], padding='SAME', scope='conv5_s1')
            conv5 = slim.max_pool2d(conv5, [3, 3], scope='pool3')
            if pool_method=='sigmoid':
                att_mask, att_logits = attentionNet(conv5, pool_method)
            else:
                att_mask = attentionNet(conv5, pool_method)
            att_map = tf.multiply(conv5, att_mask)
            att_f = tf.add(conv5, att_map)
            attended_map = tf.reduce_sum(att_f, reduction_indices=[1, 2])
            attended_map = tf.nn.l2_normalize(attended_map, dim=1)
            att_f = slim.flatten(att_f)
            fc6 = slim.fully_connected(att_f, 512, trainable=True, scope='fc6_s1')
            fc7 = slim.fully_connected(fc6, 256, activation_fn=None, trainable=True, scope='fc7_sketch')
            fc7 = tf.nn.l2_normalize(fc7, dim=1)
            final_feature_map = tf.concat(1, [fc7, attended_map])
    return final_feature_map


def init_variables(model_file):
    # pretrained_paras = ['conv1_s1', 'conv2_s1', 'conv3_s1', 'conv4_s1', 'conv5_s1', 'fc6_s1', 'fc7_sketch',
                        # 'att_conv1', 'att_conv2']
    d = np.load(model_file).item()
    pretrained_paras = d.keys()
    # print(pretrained_paras)
    init_ops = []  # a list of operations
    # for var in tf.global_variables():
    for var in tf.global_variables():
        for w_name in pretrained_paras:
            if w_name in var.name:
                # print('Initialise var %s with weight %s' % (var.name, w_name))
                init_ops.append(var.assign(d[w_name]))
    return init_ops


def load_hdf5(fname):
    hf = h5py.File(fname, 'r')
    d = {key: np.array(hf.get(key)) for key in hf.keys()}
    hf.close()
    return d


def do_singleview_crop(mat_file):
    data = loadmat(mat_file)['data']
    # crop: single view
    x = data[:, 15:15 + 225, 15:15 + 225].astype(np.float32) - 250.42
    x = x[:, :, :, np.newaxis]
    return x


def do_multiview_crop(fname, cropsize, format_flag=1):
    if format_flag == 1:
        data = loadmat(fname)['data']
    elif format_flag:
        dic = h5py.File(fname)
        data = np.array(dic['data'])
        data = data.transpose(2, 1, 0)
    if len(data.shape) == 2: # single sketch
        data = data[np.newaxis, np.newaxis, :, :]  # nxcxhxw
    elif len(data.shape) == 3: # sketch
        n, h, w = data.shape
        data = data.reshape((n, 1, h, w))
    n, c, h, w = data.shape
    xs = [0, 0, w-cropsize, w-cropsize]
    ys = [0, h-cropsize, 0, h-cropsize]
    batch_data = np.zeros((n*10, c, cropsize, cropsize), np.single)
    y_cen = int((h - cropsize) * 0.5)
    x_cen = int((w - cropsize) * 0.5)
    for i in xrange(n):
        for (k, (x, y)) in enumerate(zip(xs, ys)):
            batch_data[i*10+k, :, :, :] = data[i, :, y:y+cropsize, x:x+cropsize]
        # center crop
        batch_data[i*10+4, :, :, :] = data[i, :, y_cen:y_cen+cropsize, x_cen:x_cen+cropsize]
        for k in xrange(5):  # flip
            batch_data[i*10+k+5, :, :, :] = batch_data[i*10+k, :, :, ::-1]
    return batch_data.transpose([0, 2, 3, 1]).astype(np.float32) - 250.42


def compute_view_specific_distance(sketch_feats, image_feats):
    sketch_feats = sketch_feats.reshape([-1, NUM_VIEWS, image_feats.shape[1]])
    image_feats = image_feats.reshape([-1, NUM_VIEWS, image_feats.shape[1]])
    num = sketch_feats.shape[0]
    multi_view_dists = np.zeros((NUM_VIEWS, num, num))
    for i in xrange(NUM_VIEWS):
        # during training, we use (E-distance)^2, so here should be 'sqeuclidean'
        multi_view_dists[i,::] = ssd.cdist(sketch_feats[:, i, :], image_feats[:, i, :], 'sqeuclidean')
    return multi_view_dists


def calculate_accuracy(dist):
    top1 = 0
    top10 = 0
    for i in range(dist.shape[0]):
        rank = dist[i].argsort()
        if rank[0] == i:
            top1 = top1 + 1
        if i in rank[0:10]:
            top10 = top10 + 1
    num = dist.shape[0]
    print('top1: '+str(top1 / float(num)))
    print('top10: '+str(top10 / float(num)))
    return top1, top10


def main(_):
    subset = 'shoes'
    base_path = './data'
    im_file = '%s/%s/%s_edge_db_test.mat' % (base_path, subset, subset)
    skt_file = '%s/%s/%s_sketch_db_test.mat' % (base_path, subset, subset)
    feat_path_p = './feats/Qian_release/feats_p.mat'
    feat_path_s = './feats/Qian_release/feats_s.mat'
    dstPath = './log'
    net_model = 'deep_sbir' # 'deep_sbir' or 'DSSA'
    modelID = '%s_%s_val.txt' % (subset, net_model)
    filename = dstPath+'/'+modelID
    mean = 250.42
    model_path = './model'
    model = '%s/%s/%s/*.npy' % (model_path, subset, net_model)
    models = sorted(glob.glob(model), key=os.path.getmtime)
    for idx, model_file in enumerate(models):
        f = open(filename, 'a')
        print ("Testing model: " + model_file)
        f.write("Testing model: " + model_file)
        f.write('\n')
        # load mat file
        with tf.Graph().as_default():
            inputs = tf.placeholder(shape=[None, 225, 225, 1], dtype=tf.float32)
            if net_model == 'deep_sbir':
                net = sketch_a_net_sbir(inputs)  # construct a network
            elif net_model == 'DSSA':
                net = sketch_a_net_dssa(inputs)
            else:
                print 'Please define net_model'
            init_ops = init_variables(model_file)  # initialization
            im = do_multiview_crop(im_file, CROPSIZE)
            sketch = do_multiview_crop(skt_file, CROPSIZE)
            with tf.Session() as sess:
                sess.run(init_ops)
                im_feats = sess.run(net, feed_dict={inputs: im})
                sketch_feats = sess.run(net, feed_dict={inputs: sketch})
            # savemat(feat_path_p, {'feats': im_feats})
            # savemat(feat_path_s, {'feats': sketch_feats})
            multiview_dists = compute_view_specific_distance(sketch_feats, im_feats)
            ave_dist = multiview_dists.mean(axis=0)
            top1, top10 = calculate_accuracy(ave_dist)
            num = ave_dist.shape[0]
            f.write('top1: '+str(top1 / float(num)))
            f.write('  top10: '+str(top10 / float(num)))
            f.write('\n')
            print ("\n")
        f.close()


if __name__ == '__main__':
    print 'Evaluation metric 1: acc.@K'
    tf.app.run()
