from multiprocessing import Process, Queue
from sbir_data_util.batch_manager import MemoryBlockManager
from image_proc import Transformer
from sample_util import *


class TripletQueueRunner(object):
    def setup(self, sketch_data, image_data, sketch_label, image_label, triplets, queue_paras, phase = 'TRAIN'):
        """Setup the TripletSamplingLayer."""
        self.create_sample_fetcher(sketch_data, image_data, sketch_label, image_label, triplets, queue_paras, phase)

    def create_sample_fetcher(self, sketch_data, image_data, sketch_label, image_label, triplets, queue_paras, phase):
        self._blob_queue = Queue(10)
        self._prefetch_process = TripletSamplingDataFetcher(self._blob_queue, sketch_data, image_data, sketch_label,
                                                            image_label, triplets, queue_paras, phase)
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
    def __init__(self, queue, sketch_data, image_data, sketch_label, image_label, triplets, queue_paras, phase):
        """Setup the TripletSamplingDataLayer."""
        super(TripletSamplingDataFetcher, self).__init__()
        #        mean = mean
        self.im_size = queue_paras['im_size']
        self.cp_size = queue_paras['cp_size']
        self.chns = queue_paras['chns']
        self.mean = queue_paras['mean']
        self.batchsize = queue_paras['batch_size']
        self.num_epoch = queue_paras['num_epochs']
        self._queue = queue
        self._phase = phase
        self.sketch_transformer = Transformer(self.cp_size, self.chns, self.mean, self._phase == "TRAIN")
        if sketch_label is None or image_label is None:
            self.has_label = False
        else:
            self.has_label = True
        self.anc_bm = MemoryBlockManager(sketch_data, sketch_label, self.has_label)
        self.pos_neg_bm = MemoryBlockManager(image_data, image_label, self.has_label)
        self.triplets = triplets

    def get_next_batch_data(self):
        anc_batch = []; pos_batch = []; neg_batch = []
        # sampling
        anc_inds, pos_inds, neg_inds = zip(*sample_triplets(self.triplets, self.batchsize, self.num_epoch).next())
        #pos_inds, neg_inds = sample_triplets_trueMatch(anc_inds, self.triplets)
        # fetch data
        for (anc_id, pos_id, neg_id) in zip(anc_inds, pos_inds, neg_inds):
            anc_batch.append(self.anc_bm.get_sample(anc_id).reshape((self.im_size, self.im_size, self.chns)))
            pos_batch.append(self.pos_neg_bm.get_sample(pos_id).reshape((self.im_size, self.im_size, self.chns)))
            neg_batch.append(self.pos_neg_bm.get_sample(neg_id).reshape((self.im_size, self.im_size, self.chns)))
        # apply transform
        anc_batch = self.sketch_transformer.transform_all(anc_batch)
        pos_batch = self.sketch_transformer.transform_all(pos_batch)
        neg_batch = self.sketch_transformer.transform_all(neg_batch)
        self._queue.put((anc_batch, pos_batch, neg_batch))

    def get_next_batch_data_label(self):
        anc_batch = []; pos_batch = []; neg_batch = []
        # sampling
        anc_inds, pos_inds, neg_inds = zip(*sample_triplets(self.triplets, self.batchsize, self.num_epoch).next())
        #pos_inds, neg_inds = sample_triplets_trueMatch(anc_inds, self.triplets)
        # fetch data
        for (anc_id, pos_id, neg_id) in zip(anc_inds, pos_inds, neg_inds):
            anc_batch.append(self.anc_bm.get_sample(anc_id))
            pos_batch.append(self.pos_neg_bm.get_sample(pos_id))
            neg_batch.append(self.pos_neg_bm.get_sample(neg_id))
        # apply transform
        anc_batch_data, anc_batch_label = self.sketch_transformer.transform_all_with_label(anc_batch)
        pos_batch_data, pos_batch_label = self.sketch_transformer.transform_all_with_label(pos_batch)
        neg_batch_data, neg_batch_label = self.sketch_transformer.transform_all_with_label(neg_batch)
        self._queue.put((anc_batch_data, pos_batch_data, neg_batch_data, anc_batch_label, pos_batch_label, neg_batch_label))

    def run(self):
        print 'TripletSamplingDataFetcher started'
        if self.has_label:
            while True:
                self.get_next_batch_data_label()
        else:
            while True:
                self.get_next_batch_data()


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