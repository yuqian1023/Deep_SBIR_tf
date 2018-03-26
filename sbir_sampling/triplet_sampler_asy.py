from multiprocessing import Process, Queue
from sbir_util.batch_manager import MemoryBlockManager
from image_proc import Transformer
from sample_util import *


class TripletSamplingLayer(object):
    def setup(self, sketch_dir, image_dir, triplet_path, mean, hard_ratio, batch_size, phase):
        """Setup the TripletSamplingLayer."""
        self.create_sample_fetcher(sketch_dir, image_dir, triplet_path, mean, hard_ratio, batch_size, phase)

    def create_sample_fetcher(self, sketch_dir, image_dir, triplet_path, mean, hard_ratio, batch_size, phase):
        self._blob_queue = Queue(10)
        self._prefetch_process = TripletSamplingDataFetcher(self._blob_queue, sketch_dir, image_dir, triplet_path, mean, hard_ratio, batch_size, phase)
        self._prefetch_process.start()
        def cleanup():
            print 'Terminating BlobFetcher'
            self._prefetch_process.terminate()
            self._prefetch_process.join()
        import atexit
        atexit.register(cleanup)

    def get_next_batch(self):
        return self._blob_queue.get()


class TripletSamplingDataFetcher(Process):
    def __init__(self, queue, sketch_dir, image_dir, triplet_path, mean, hard_ratio, batch_size, phase):
        """Setup the TripletSamplingDataLayer."""
        super(TripletSamplingDataFetcher, self).__init__()
        #        mean = mean
        self._queue = queue
        self._phase = phase
        self.sketch_transformer = Transformer(225, 1, mean, self._phase == "TRAIN")
        self.sketch_dir = sketch_dir
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
        if 'handbags' in self.sketch_dir:
            # positive are always true match
            pos_inds, neg_inds = sample_triplets_trueMatch(anc_inds, self._phase)
        else:
            pos_inds, neg_inds = sample_triplets_pos_neg(anc_inds, self.triplets, self.neg_list, self.hard_ratio)

        # fetch data
        for (anc_id, pos_id, neg_id) in zip(anc_inds, pos_inds, neg_inds):
            anc_batch.append(self.anc_bm.get_sample(anc_id).reshape((256, 256, 1)))
            pos_batch.append(self.pos_neg_bm.get_sample(pos_id).reshape((256, 256, 1)))
            neg_batch.append(self.pos_neg_bm.get_sample(neg_id).reshape((256, 256, 1)))
        # apply transform
        anc_batch = self.sketch_transformer.transform_all(anc_batch).astype(np.uint8)
        pos_batch = self.sketch_transformer.transform_all(pos_batch).astype(np.uint8)
        neg_batch = self.sketch_transformer.transform_all(neg_batch).astype(np.uint8)
        self._queue.put((anc_batch, pos_batch, neg_batch))

    def run(self):
        print 'TripletSamplingDataFetcher started'
        while True:
            self.get_next_batch()


if __name__ == '__main__':
    print 'TripletSamplingDataFetcher started'
    sketch_dir = './shoes_sketch_db_train.mat'
    image_dir = './shoes_edge_db_train.mat'
    triplet_path = './shoes_annotation.json'
    mean = 250.42
    hard_ratio = 0.75
    batch_size = 16
    phase = 'TRAIN'
    triplet_sampler = TripletSamplingLayer()
    triplet_sampler.setup(sketch_dir, image_dir, triplet_path, mean, hard_ratio, batch_size, phase)
    while True:
        triplet_sampler.get_next_batch()