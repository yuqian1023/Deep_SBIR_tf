from multiprocessing import Process, Queue
from image_proc import Transformer
from sbir_util.batch_manager import MemoryBlockManager
from image_proc import Transformer
from sample_util import *
from scipy.io import loadmat


class TripletSamplingLayer(object):
    def __init__(self, sketch_dir, image_dir, triplet_path, mean, hard_ratio, batch_size, phase):
        """Setup the TripletSamplingDataLayer."""
        self._queue = Queue(10)
#        mean = mean
        self._phase = phase
        self.sketch_transformer = Transformer(225, 1, mean, self._phase == "TRAIN")
        self.anc_bm = MemoryBlockManager(sketch_dir)
        self.pos_neg_bm = MemoryBlockManager(image_dir)
        self.hard_ratio = hard_ratio
        self.mini_batchsize = batch_size
        self.load_triplets(triplet_path)

    def load_triplets(self, triplet_path):
        self.triplets, self.neg_list = load_triplets(triplet_path, self._phase)

    def get_next_batch(self):
        anc_batch = []; pos_batch = []; neg_batch = []
        # sampling
        anc_inds = self.anc_bm.pop_batch_inds_circular(self.mini_batchsize)
        pos_inds, neg_inds = sample_triplets(anc_inds, self.triplets, self.neg_list, self.hard_ratio)
        # fetch data
        for (anc_id, pos_id, neg_id) in zip(anc_inds, pos_inds, neg_inds):
            anc_batch.append(self.anc_bm.get_sample(anc_id).reshape((256, 256, 1)))
            pos_batch.append(self.pos_neg_bm.get_sample(pos_id).reshape((256, 256, 1)))
            neg_batch.append(self.pos_neg_bm.get_sample(neg_id).reshape((256, 256, 1)))
        # apply transform
        anc_batch = self.sketch_transformer.transform_all(anc_batch)
        pos_batch = self.sketch_transformer.transform_all(pos_batch)
        neg_batch = self.sketch_transformer.transform_all(neg_batch)
        # self._queue.put((anc_batch, pos_batch, neg_batch))
        return anc_batch, pos_batch, neg_batch

if __name__ == '__main__':
    print 'TripletSamplingDataFetcher started'
    sketch_dir = './shoes_sketch_db_train.mat'
    image_dir = './shoes_edge_db_train.mat'
    triplet_path = './shoes_annotation.json'
    mean = 250.42
    hard_ratio = 0.75
    batch_size = 128
    phase = 'TRAIN'
    triplet_sampler = TripletSamplingLayer(sketch_dir, image_dir, triplet_path, mean, hard_ratio, batch_size, phase)
    while True:
        triplet_sampler.get_next_batch()
