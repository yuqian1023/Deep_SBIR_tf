import tensorflow as tf
from tensorflow.contrib import slim
import os


def mkdir_if_missing(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def expand_x_y_dims(f):
    return tf.expand_dims(tf.expand_dims(f, 1), 1)


def _get_tensor_shape(x):
    s = x.get_shape().as_list()
    return [i if i is not None else -1 for i in s]


def _get_tensor_rank(x):
    s = x.get_shape().as_list()
    return len(s)


def _is_4d_tensor(x):
    return len(_get_tensor_shape(x)) == 4


def reshape_feats(x, mode='spatial'):
    fm_shape = _get_tensor_shape(x)
    import pdb
    pdb.set_trace()
    if mode=='spatial':
        num_rows = fm_shape[1] ** 2
        num_channels = fm_shape[3]
        # transpose feature map
        fm = tf.reshape(x, shape=[-1, num_rows, num_channels])
    return fm


def spatial_softmax(fm):
    fm_shape = _get_tensor_shape(fm)
    n_grids = fm_shape[1] ** 2
    # transpose feature map
    fm = tf.transpose(fm, perm=[0, 3, 1, 2])
    t_fm_shape = _get_tensor_shape(fm)
    fm = tf.reshape(fm, shape=[-1, n_grids])
    # apply softmax
    prob = tf.nn.softmax(fm)
    # reshape back
    prob = tf.reshape(prob, shape=t_fm_shape)
    prob = tf.transpose(prob, perm=[0, 2, 3, 1])
    return prob


def soft_attention_pool(im, im_ctx):
    att_logits = slim.conv2d(im_ctx, 1, [1, 1],
                             activation_fn=None,
                             scope='att_logits')
    att = spatial_softmax(att_logits)
    im_att = tf.mul(att, im)
    return tf.reduce_sum(im_att, reduction_indices=[1, 2])


def mm_conv_concat(im, ctx, embed_dim, scope=""):
    '''
    Multi-Modals fusion, embed to the same dimensionality and then concat
    im: image (conv5/pool5)
    ctx: domain label (n*2) binary
    embed_dim: context embeding dim: 32
    '''
    scope = scope or "mbp"
    with tf.variable_scope(scope):
        ctx_emb = slim.fully_connected(ctx, embed_dim, scope='ctx_emb')
        assert (_is_4d_tensor(im))
        ctx_emb = expand_x_y_dims(ctx_emb)
        _, w, h, _ = _get_tensor_shape(im)
        ctx_emb = tf.tile(ctx_emb, [1, w, h, 1])
        embed = tf.concat(concat_dim=3, values=[im, ctx_emb])
        # embed = slim.dropout(embed, keep_prob=keep_prob)
    return embed