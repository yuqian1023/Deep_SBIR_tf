import numpy as np
import numpy.random as nr
from sbir_util.smts_api import SMTSApi
from itertools import izip_longest as zip_longest
from util import alternate


def sample_triplets(anc_inds, triplets, neg_list, hard_ratio, phase):
    pos_inds = []
    neg_inds = []
    for anc_id in anc_inds:
        tuples = triplets[anc_id]
        if phase.lower() == 'train':
            key_num = 65535
            pos_id = 65535
        else:
            key_num = 255
            pos_id = 255
        if phase.lower() == 'train':
            ptr = 0
            while (pos_id == key_num or neg_id == key_num):
                idx = nr.randint(len(tuples))
                pos_id, neg_id = tuples[idx]
                ptr += 1
            pos_inds.append(pos_id)
        else:
            idx = 0
            # idx = nr.randint(len(tuples))
            # pos_id, neg_id = tuples[idx]
            pos_id = anc_id
            while True:
                neg_id = nr.randint(239)
                if neg_id != pos_id:
                    break
            pos_inds.append(pos_id)
        # import pdb
        # pdb.set_trace()
        if nr.rand() > hard_ratio:  # sample easy
            nidx = nr.randint(neg_list.shape[1])
            neg_id = neg_list[anc_id, nidx]
        neg_inds.append(neg_id)
    return pos_inds, neg_inds


def sample_triplets_with_filter_bk(anc_inds, triplets, bbox, neg_list, hard_ratio, phase):
    pos_inds = []
    neg_inds = []
    bbox_inds = []
    for anc_id in anc_inds:
        tuples = triplets[anc_id]
        if phase.lower() == 'train':
            key_num = 65535
            pos_id = 65535
        else:
            key_num = 255
            pos_id = 255
        if phase.lower() == 'train':
            while (pos_id == key_num or neg_id == key_num):
                idx = nr.randint(len(tuples))
                pos_id, neg_id = tuples[idx]
            pos_inds.append(pos_id)
        else:
            idx = 0
            pos_id = anc_id
            while True:
                neg_id = nr.randint(239)
                if neg_id != pos_id:
                    break
            pos_inds.append(pos_id)
        if nr.rand() > hard_ratio:  # sample easy
            # print 'Wow, so easy!'
            nidx = nr.randint(neg_list.shape[1])
            neg_id = neg_list[anc_id, nidx]
        neg_inds.append(neg_id)
        bbox_inds.append(bbox[anc_id][idx])
    return pos_inds, neg_inds, bbox_inds


def sample_triplets_with_filter(anc_inds, triplets, bbox, neg_list, hard_ratio, phase,
                                use_weighting=False):
    # print 'sampling'
    pos_inds = []
    neg_inds = []
    bbox_inds = []
    batch_size = len(anc_inds)
    weights = np.ones((batch_size,), dtype=np.float32)

    for i, anc_id in enumerate(anc_inds):
        tuples = triplets[anc_id]
        if phase.lower() == 'train':
            key_num = 65535
            pos_id = 65535
        else:
            key_num = 255
            pos_id = 255
        if phase.lower() == 'train':
            while (pos_id == key_num or neg_id == key_num):
                idx = nr.randint(len(tuples))
                pos_id, neg_id = tuples[idx]
                box_tmp = bbox[anc_id][idx]
            pos_inds.append(pos_id)

        else:
            idx = 0
            pos_id = anc_id
            while True:
                neg_id = nr.randint(239)
                if neg_id != pos_id:
                    break
            pos_inds.append(pos_id)
        if nr.rand() > hard_ratio:  # sample easy
            # print 'Wow, so easy!'
            weights[i] = 0.
            nidx = nr.randint(neg_list.shape[1])
            neg_id = neg_list[anc_id, nidx]
            box_tmp = [[1.0, 1.0, 225.0, 225.0, 1.0, 1.0, 225.0, 225.0, 1.0, 1.0, 225.0, 225.0, 1.0, 1.0, 225.0, 225.0, 1.0, 1.0, 225.0, 225.0, 1.0, 1.0, 225.0, 225.0]]
        neg_inds.append(neg_id)
        bbox_inds.append(box_tmp)
    if use_weighting:
        # print len(bbox_inds)
        # print len(bbox_inds[0][0])
        return pos_inds, neg_inds, bbox_inds, weights
    else:
        return pos_inds, neg_inds, bbox_inds



def sample_triplets_with_filter_refined(anc_inds, triplets, bbox, neg_list, hard_ratio, phase):
    pos_inds = []
    neg_inds = []
    bbox_inds = []
    for anc_id in anc_inds:
        tuples = triplets[anc_id]
        if phase.lower() == 'train':
            key_num = 65535
            pos_id = 65535
        else:
            key_num = 255
            pos_id = 255
        while (pos_id == key_num or neg_id == key_num):
            idx = nr.randint(len(tuples))
            pos_id, neg_id = tuples[idx]
        pos_inds.append(pos_id)
        if nr.rand() > hard_ratio:  # sample easy
            # print 'Wow, so easy!'
            nidx = nr.randint(neg_list.shape[1])
            neg_id = neg_list[anc_id, nidx]
        neg_inds.append(neg_id)
        bbox_inds.append(bbox[anc_id][idx])
    return pos_inds, neg_inds, bbox_inds


def sample_triplets_pos_neg(anc_inds, triplets, neg_list, hard_ratio):
    pos_inds = []
    neg_inds = []
    for anc_id in anc_inds:
        tuples = triplets[anc_id]
        idx = nr.randint(len(tuples))
        pos_id, neg_id = tuples[idx]
        pos_inds.append(pos_id)
        # import pdb
        # pdb.set_trace()
        if nr.rand() > hard_ratio:  # sample easy
            nidx = nr.randint(neg_list.shape[1])
            neg_id = neg_list[anc_id, nidx]
        neg_inds.append(neg_id)
    return pos_inds, neg_inds



def sample_triplets_trueMatch(anc_inds, phase):
    pos_inds = []
    neg_inds = []
    for anc_id in anc_inds:
        pos_id = anc_id
        pos_inds.append(pos_id)
        # all_inds = np.unique(triplets)
        if phase=="TRAIN":
            all_inds = range(0,400)
        else:
            all_inds = range(0,168)
        neg_list = np.setdiff1d(all_inds, anc_id)
        neg_id = np.random.choice(neg_list)
        neg_inds.append(neg_id)
    return pos_inds, neg_inds


def load_triplets(triplet_path, subset):
    smts_api = SMTSApi(triplet_path)
    triplets = smts_api.get_triplets(subset)
    return triplets, make_negative_list(triplets)


def load_triplets_bbox(triplet_path, subset):
    smts_api = SMTSApi(triplet_path)
    triplets, bbox = smts_api.get_triplets_bbox(subset)
    return triplets, make_negative_list_bbox(triplets, subset), bbox


def make_negative_list(triplets):
    tri_mat = np.array(triplets)
    num_images = tri_mat.shape[0]
    all_inds = np.unique(triplets)
    neg_list = []
    for i in xrange(num_images):
        pos_inds = np.union1d(tri_mat[i, :, 0], tri_mat[i, :, 1])
        neg_inds = np.setdiff1d(all_inds, pos_inds).reshape([1, -1])
        neg_list.append(neg_inds)
    return np.concatenate(neg_list).astype(np.int32)


def make_negative_list_bbox(triplets, subset):
    tri_mat = np.array(triplets)
    num_images = tri_mat.shape[0]
    all_inds = np.unique(triplets)
    if subset.lower() == 'train':
        key_num = 65535
        max_len = 1761
    else:
        key_num = 255
        max_len = 239
    if key_num in all_inds:
        index = np.argwhere(all_inds==key_num)
        all_inds = np.delete(all_inds, index)
    neg_list = []
    for i in xrange(num_images):
        pos_inds = np.union1d(tri_mat[i, :, 0], tri_mat[i, :, 1])
        if key_num in pos_inds:
            index = np.argwhere(pos_inds == key_num)
            pos_inds = np.delete(pos_inds, index)
        neg_inds = np.setdiff1d(all_inds, pos_inds).reshape([1, -1])
        if neg_inds.shape[1] < max_len:
            diff = max_len - neg_inds.shape[1]
            neg_inds = np.append(neg_inds, np.repeat(neg_inds[:,-1],diff))
        neg_list.append(neg_inds)
    # return np.concatenate(neg_list).astype(np.int32)
    return np.array(neg_list)