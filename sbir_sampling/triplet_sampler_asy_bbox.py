from multiprocessing import Process, Queue
import pdb
from sbir_util_original.batch_manager import MemoryBlockManager
from image_proc import Transformer
from sample_util import *
import pylab as plt
from scipy.io import loadmat


class TripletSamplingLayer(object):
    def setup(self, sketch_dir, image_dir, triplet_path, mean, hard_ratio, batch_size, phase, mode, att_flag=False):
        """Setup the TripletSamplingLayer."""
        self.create_sample_fetcher(sketch_dir, image_dir, triplet_path, mean, hard_ratio, batch_size, phase, mode,
                                   att_flag)

    def create_sample_fetcher(self, sketch_dir, image_dir, triplet_path, mean, hard_ratio, batch_size, phase, mode,
                              att_flag):
        self._blob_queue = Queue(10)
        self._prefetch_process = []
        self._num_proc = 1
        for i in range(self._num_proc):
            proc = TripletSamplingDataFetcher(self._blob_queue, sketch_dir, image_dir, triplet_path, mean,
                                              hard_ratio, batch_size, phase, i, mode, att_flag)
            proc.start()
            self._prefetch_process.append(proc)

        def cleanup():
            print 'Terminating BlobFetcher'
            for pid in range(self._num_proc):
                self._prefetch_process[pid].terminate()
                self._prefetch_process[pid].join()

        import atexit
        atexit.register(cleanup)

    def get_next_batch(self):
        return self._blob_queue.get()


class TripletSamplingDataFetcher(Process):
    def __init__(self, queue, sketch_dir, image_dir, triplet_path, mean, hard_ratio, batch_size, phase, proc_id, mode,
                 att_flag=False):
        """Setup the TripletSamplingDataLayer."""
        super(TripletSamplingDataFetcher, self).__init__()
        #        mean = mean
        self._queue = queue
        self._phase = phase
        self.proc_id = proc_id
        self.sketch_transformer = Transformer(225, 1, mean, self._phase == "TRAIN")
        self.anc_bm = MemoryBlockManager(sketch_dir)
        self.disable_anc_id = 1484
        self.valid_anc_inds = np.setdiff1d(np.arange(self.anc_bm.num_samples), np.array(self.disable_anc_id))
        self.pos_neg_bm = MemoryBlockManager(image_dir)
        self.hard_ratio = hard_ratio
        self.mode = mode
        self.mini_batchsize = batch_size
        self.att_flag = att_flag
        self.load_triplets(triplet_path)

        # self.tmp_flag = True
        # self.idx = 0
        print 'Done'

    def load_triplets(self, triplet_path):
        if self.mode == 'whole-part' or self.mode == 'part':
            self.triplets, self.neg_list, self.bbox = load_triplets_bbox(triplet_path, self._phase)
        else:
            self.triplets, self.neg_list = load_triplets(triplet_path, self._phase)
        if self.att_flag:
            print 'load attribute annotations'
            mat_file = '/homes/qian/libsvm/libsvm-3.18/Project_2/Base_Function/step2-attribute/mat_file/attribute_annotation.mat'
            if self._phase == 'TRAIN':
                self.att = loadmat(mat_file)['att_trn'][:, 0:8]
            else:
                self.att = loadmat(mat_file)['att_tst'][:, 0:8]

    def get_next_batch(self):
        # print 'Start fetching batch %d' % (self.idx)
        anc_batch = [];
        pos_batch = [];
        neg_batch = []
        if self.att_flag:
            att_anc_list = [];
            att_pos_list = [];
            att_neg_list = []

        # sampling
        anc_inds = self.anc_bm.pop_batch_inds_circular(self.mini_batchsize)
        anc_inds = np.array(anc_inds)

        # in order to avoid the sample whose triplet pairs are all overlapped with another set,
        # we remove such samples from anchor list, thus we don't need to check when smapling triplets
        if np.any(anc_inds == self.disable_anc_id):
            new_id = np.random.choice(self.valid_anc_inds, size=(1,))
            anc_inds[anc_inds == self.disable_anc_id] = new_id[0]

        anc_inds = anc_inds.tolist()

        if self._phase == 'TRAIN' and (self.mode == 'whole-part' or self.mode == 'part'):
            # print('LOG1')
            pos_inds, neg_inds, bbox_inds, weights = sample_triplets_with_filter(anc_inds, self.triplets, self.bbox,
                                                                                 self.neg_list, self.hard_ratio,
                                                                                 self._phase,
                                                                                 use_weighting=True)
        else:
            # print('LOG2')
            try:
                pos_inds, neg_inds = sample_triplets(anc_inds, self.triplets, self.neg_list, self.hard_ratio,
                                                     self._phase)
            except Exception, e:
                print(str(e))
                raise e
                # print('whole')

        # fetch data
        for (anc_id, pos_id, neg_id) in zip(anc_inds, pos_inds, neg_inds):
            anc_batch.append(self.anc_bm.get_sample(anc_id).reshape((256, 256, 1)))
            pos_batch.append(self.pos_neg_bm.get_sample(pos_id).reshape((256, 256, 1)))
            neg_batch.append(self.pos_neg_bm.get_sample(neg_id).reshape((256, 256, 1)))
            if self.att_flag:
                att_anc_list.append(self.att[anc_id, :])
                att_pos_list.append(self.att[pos_id, :])
                att_neg_list.append(self.att[neg_id, :])

        # apply transform
        anc_batch_trans = self.sketch_transformer.transform_all_with_bbox(anc_batch, bbox_inds, 'anc')
        pos_batch_trans = self.sketch_transformer.transform_all_with_bbox(pos_batch, bbox_inds, 'pos')
        neg_batch_trans = self.sketch_transformer.transform_all_with_bbox(neg_batch, bbox_inds, 'neg')
        # print 'finish transforming'
        # anc_batch_trans = self.sketch_transformer.transform_all(anc_batch)
        # pos_batch_trans = self.sketch_transformer.transform_all(pos_batch)
        # neg_batch_trans = self.sketch_transformer.transform_all(neg_batch)

        # if self._phase == 'TRAIN' and (self.mode=='whole-part' or self.mode=='part'):
        #     anc_batch_part1, anc_batch_part2 = self.sketch_transformer.transform_all_part(anc_batch, bbox_inds, 'anc')
        #     pos_batch_part1, pos_batch_part2 = self.sketch_transformer.transform_all_part(pos_batch, bbox_inds, 'pos')
        #     neg_batch_part1, neg_batch_part2 = self.sketch_transformer.transform_all_part(neg_batch, bbox_inds, 'neg')
        #     if self.mode=='whole-part':
        #         anc_batch = np.concatenate((anc_batch_trans, anc_batch_part1, anc_batch_part2),axis=0)
        #         pos_batch = np.concatenate((pos_batch_trans, pos_batch_part1, pos_batch_part2),axis=0)
        #         neg_batch = np.concatenate((neg_batch_trans, neg_batch_part1, neg_batch_part2),axis=0)
        #     else:
        #         anc_batch = np.concatenate((anc_batch_part1, anc_batch_part2),axis=0)
        #         pos_batch = np.concatenate((pos_batch_part1, pos_batch_part2),axis=0)
        #         neg_batch = np.concatenate((neg_batch_part1, neg_batch_part2),axis=0)
        #     if self.att_flag:
        #         dim_att = self.att.shape[1]
        #         att_anc_batch = np.reshape(att_anc_list, (len(anc_inds), dim_att))
        #         att_pos_batch = np.reshape(att_pos_list, (len(anc_inds), dim_att))
        #         att_neg_batch = np.reshape(att_neg_list, (len(anc_inds), dim_att))
        #         self._queue.put((anc_batch, pos_batch, neg_batch, att_anc_batch, att_pos_batch, att_neg_batch))
        #     else:
        #         self._queue.put((anc_batch, pos_batch, neg_batch))
        # else:
        if self.att_flag:
            dim_att = self.att.shape[1]
            att_anc_batch = np.reshape(att_anc_list, (len(anc_inds), dim_att))
            att_pos_batch = np.reshape(att_pos_list, (len(anc_inds), dim_att))
            att_neg_batch = np.reshape(att_neg_list, (len(anc_inds), dim_att))
            self._queue.put(
                (anc_batch_trans, pos_batch_trans, neg_batch_trans, att_anc_batch, att_pos_batch, att_neg_batch))
        else:
            self._queue.put((anc_batch_trans, pos_batch_trans, neg_batch_trans, weights))
            # print 'finish sampling'

            # self.idx += 1

    def run(self):
        print 'TripletSamplingDataFetcher started'
        np.random.seed(self.proc_id)
        while True:
            self.get_next_batch()


def vis_batch(anc, pos, neg, n_vis=1):
    num = anc.shape[0]
    sample_inds = np.arange(num)
    np.random.shuffle(sample_inds)
    vis_inds = sample_inds[:n_vis]

    def get_image(data, sample_id, part_id):
        _idx = sample_id + part_id * num
        im = np.tile(data[_idx], [1, 1, 3])
        return im.astype(np.uint8).copy()

    data_type = ['anchor', 'pos', 'neg']
    batch_data = [anc, pos, neg]
    for i in range(n_vis):
        idx = vis_inds[i]
        for bid, batch in enumerate(batch_data):
            # print(batch.mean())
            for part_id in range(2):
                im = get_image(batch, idx, part_id)
                plt.subplot(3, 3, bid * 3 + part_id + 1)
                plt.imshow(im)
                plt.title('%s: p%d' % (data_type[bid], part_id))
        plt.show()


def vis_batch_part(anc, pos, neg, n_vis=5):
    num = anc.shape[0] / 2
    sample_inds = np.arange(num)
    np.random.shuffle(sample_inds)
    vis_inds = sample_inds[:n_vis]

    def get_image(data, sample_id, part_id):
        _idx = sample_id + part_id * num
        im = np.tile(data[_idx], [1, 1, 3])
        return im.astype(np.uint8).copy()

    data_type = ['anchor', 'pos', 'neg']
    batch_data = [anc, pos, neg]
    for i in range(n_vis):
        idx = vis_inds[i]
        for bid, batch in enumerate(batch_data):
            # print(batch.mean())
            for part_id in range(2):
                im = get_image(batch, idx, part_id)

                plt.subplot(3, 2, bid * 2 + part_id + 1)
                plt.imshow(im)
                plt.title('%s: p%d' % (data_type[bid], part_id))
        plt.show()


def vis_batch_whole(anc, pos, neg, n_vis=5):
    import pylab as plt
    num = anc.shape[0] / 1
    sample_inds = np.arange(num)
    np.random.shuffle(sample_inds)
    vis_inds = sample_inds[:n_vis]

    def get_image(data, sample_id, part_id):
        _idx = sample_id + part_id * num
        im = np.tile(data[_idx], [1, 1, 3])
        return im.astype(np.uint8).copy()

    data_type = ['anchor', 'pos', 'neg']
    batch_data = [anc, pos, neg]
    for i in range(n_vis):
        idx = vis_inds[i]
        for bid, batch in enumerate(batch_data):
            # print(batch.mean())
            for part_id in range(1):
                im = get_image(batch, idx, part_id)

                plt.subplot(3, 1, bid * 1 + part_id + 1)
                plt.imshow(im)
                plt.title('%s: p%d' % (data_type[bid], part_id))
        plt.show()


def vis_batch_whole_mask(anc, pos, neg, n_vis=5):
    import pylab as plt
    anc, anc_mask = np.split(anc, 2, axis=3)
    pos, pos_mask = np.split(pos, 2, axis=3)
    neg, neg_mask = np.split(neg, 2, axis=3)

    print ('%0.3f, %0.3f, %0.3f' % (anc_mask.max(),
                                    pos_mask.max(),
                                    neg_mask.max()))
    print ('%0.3f, %0.3f, %0.3f' % (anc_mask.min(),
                                    pos_mask.min(),
                                    neg_mask.min()))

    from scipy.misc import imresize as resize
    def _proc_mask(masks):
        new_masks = []
        for m in masks:
            m = np.squeeze(m, 2)
            print(m.max())
            down_m = resize(m, size=[7, 7])
            print(down_m.max())
            up_m = resize(down_m, [225, 225])
            print(up_m.max())
            new_masks.append(up_m[np.newaxis, :, :, np.newaxis])
        print(new_masks[-1].shape)
        return np.concatenate(new_masks, axis=0) / 255.

    anc *= _proc_mask(anc_mask)
    pos *= _proc_mask(pos_mask)
    neg *= _proc_mask(neg_mask)

    # anc_mask = pos_mask = neg_mask = 1.
    print ('%0.3f, %0.3f, %0.3f' % (anc.max(),
                                    pos.max(),
                                    neg.max()))

    num = anc.shape[0] / 1
    sample_inds = np.arange(num)
    np.random.shuffle(sample_inds)
    vis_inds = sample_inds[:n_vis]

    def get_image(data, sample_id, part_id):
        _idx = sample_id + part_id * num
        im = np.tile(data[_idx], [1, 1, 3])
        return im.astype(np.uint8).copy()

    data_type = ['anchor', 'pos', 'neg']
    batch_data = [anc, pos, neg]
    for i in range(n_vis):
        idx = vis_inds[i]
        for bid, batch in enumerate(batch_data):
            # print(batch.mean())
            for part_id in range(1):
                im = get_image(batch, idx, part_id)
                plt.subplot(3, 1, bid * 1 + part_id + 1)
                plt.imshow(im)
                plt.title('%s: p%d' % (data_type[bid], part_id))
        plt.show()


def vis_batch_with_att(anc, pos, neg, att, n_vis=5):
    import pylab as plt
    num = anc.shape[0] / 1
    sample_inds = np.arange(num)
    np.random.shuffle(sample_inds)
    vis_inds = sample_inds[:n_vis]

    def get_image(data, sample_id, part_id):
        _idx = sample_id + part_id * num
        im = np.tile(data[_idx], [1, 1, 3])
        return im.astype(np.uint8).copy()

    data_type = ['anchor', 'pos', 'neg']
    batch_data = [anc, pos, neg]
    for i in range(n_vis):
        idx = vis_inds[i]
        for bid, batch in enumerate(batch_data):
            # print(batch.mean())
            for part_id in range(1):
                im = get_image(batch, idx, part_id)

                plt.subplot(3, 1, bid * 1 + part_id + 1)
                plt.imshow(im)
                plt.title('%s: p%d' % (data_type[bid], part_id))
        plt.show()


if __name__ == '__main__':
    print 'TripletSamplingDataFetcher started'
    subset = 'shoes2K-v2'
    mean = 250.42
    hard_ratio = 1.0
    batch_size = 32
    phase = 'TRAIN'
    phase_te = 'TEST'
    mode = 'whole-part'
    att_flag = False
    sketch_dir = '/import/vision-ephemeral/QY/code/triplet_with_bbox/data/%s/%s_sketch_db_%s.mat' % (
    subset, subset, phase.lower())
    image_dir = '/import/vision-ephemeral/QY/code/triplet_with_bbox/data/%s/%s_edge_db_%s.mat' % (
    subset, subset, phase.lower())
    triplet_path = '/import/vision-ephemeral/QY/code/triplet_with_bbox/data/%s/%s_annotation.json' % (subset, subset)
    sketch_dir_te = '/import/vision-ephemeral/QY/code/triplet_with_bbox/data/%s/%s_sketch_db_%s.mat' % (
    subset, subset, phase_te.lower())
    image_dir_te = '/import/vision-ephemeral/QY/code/triplet_with_bbox/data/%s/%s_edge_db_%s.mat' % (
    subset, subset, phase_te.lower())
    triplet_sampler = TripletSamplingLayer()
    triplet_sampler.setup(sketch_dir, image_dir, triplet_path, mean, hard_ratio, batch_size, phase, mode, att_flag)

    if att_flag:
        anc, pos, neg, att_anc, att_pos, att_neg = triplet_sampler.get_next_batch()
    else:
        anc, pos, neg = triplet_sampler.get_next_batch()

        if mode == 'whole-part':
            # anc_whole, anc_bbox = np.split(anc, 2, axis=3)
            # pos_whole, pos_bbox = np.split(pos, 2, axis=3)
            # neg_whole, neg_bbox = np.split(neg, 2, axis=3)
            # vis_batch_whole(anc_whole, pos_whole, neg_whole, 3)
            # vis_batch_whole(anc_bbox, pos_bbox, neg_bbox, 3)
            vis_batch_whole_mask(anc, pos, neg, 10)
            # vis_batch(anc_bbox, pos_bbox, neg_bbox, 3)
        elif mode == 'part':
            vis_batch_part(anc, pos, neg, 3)
        else:
            vis_batch_whole(anc, pos, neg, 3)
