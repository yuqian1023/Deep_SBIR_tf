import numpy as np
import numpy.random as nr
from matplotlib import pyplot as plt
try:
    from skimage.transform import rotate, resize
except:
    print 'warning: skimage not installed, disable rotation'


def rand_rotate(im, rotate_amp):
    deg = (2 * nr.rand() - 1) * rotate_amp
    # print deg
    rot_im = rotate(im, deg, mode='nearest') * 255
    return rot_im.astype(np.uint8)


def is_color(im):
    if len(im.shape) == 3:
        if im.shape[-1] == 3:
            return True
        else:
            return False
    else:
        return False


def imshow(im):
    im = im.astype(np.uint8)
    if not is_color(im):
        im = im.reshape((im.shape[0], im.shape[1], 1))
        im = np.tile(im, (1, 1, 3))
    else:
        im = im[:, :, ::-1]  # switch to RGB
    #pl.imshow(im)
    #pl.show()


def deprocess(data_pack, mean):
    crop_size = data_pack.shape[-1]
    mean_size = mean.shape[0]
    isclr = is_color(mean)
    imshow(mean)
    if mean_size > crop_size:
        x = (mean_size - crop_size) / 2
        if isclr:
            mean = mean[x:x+crop_size, x:x+crop_size, :]
        else:
            mean = mean[x:x+crop_size, x:x+crop_size]
    imshow(mean)
    batchsize = data_pack.shape[0]
    for i in xrange(batchsize):
        elem = data_pack[i, :, :, :]
        elem = elem.transpose((1, 2, 0)) + mean
        elem[elem > 255] = 255
        elem[elem < 0] = 0
        imshow(elem.astype(np.uint8))


def undo_trans(data, mean):
    data += mean
    data = data.transpose((1, 2, 0))
    if data.shape[-1] == 1:
        data = np.tile(data, (1, 1, 3))
    else:
        data = data[:, :, ::-1]
    print '%0.2f %0.2f' % (data.max(), data.min())
    return data.astype(np.uint8)


class Transformer:
    def __init__(self, crop_size, num_channels, mean_=None, is_train=False, rotate_amp=None):
        self._crop_size = crop_size
        self._in_size = 256
        self._boarder_size = self._in_size - self._crop_size
        self._num_channels = num_channels
        self._is_train = is_train
        self._rotate_amp = rotate_amp
        if self._num_channels > 1 and self._rotate_amp > 0:
            raise Exception("can not rotate color image")
        if type(mean_) == str:
            mean_mat = np.load(mean_)
            self._mean = mean_mat.mean(axis=-1).mean(axis=-1).reshape(1, 3, 1, 1)  # mean value
        else:
            self._mean = mean_

    # @profile
    def transform(self, im):
        if len(im.shape) == 1:
            im = im.reshape((self._in_size, self._in_size)) if (self._num_channels == 1) else \
                im.reshape((self._num_channels, self._in_size, self._in_size))
        # rotation
        im1 = rand_rotate(im, self._rotate_amp) if self._rotate_amp is not None else im

        # translation and flip
        if len(im1.shape) == 2:  # gray scale
            im1 = im1.reshape((1, im1.shape[0], im1.shape[1]))
        x = nr.randint(self._boarder_size) if self._is_train else self._boarder_size / 2
        y = nr.randint(self._boarder_size) if self._is_train else self._boarder_size / 2

        if nr.random() > 0.5 and self._is_train:
            im2 = im1[:, y:y+self._crop_size, x+self._crop_size:x:-1]
        else:
            im2 = im1[:, y:y+self._crop_size, x:x+self._crop_size]
        return im2

    def transform_all(self, imlist):
        processed = []
        for im in imlist:
            if im.shape[-1] == self._crop_size:
                processed.append(im.reshape(1, self._crop_size, self._crop_size, self._num_channels))
                continue
            # translation and flip for image
            im = im.reshape(1, self._in_size, self._in_size, self._num_channels)
            x = nr.randint(self._boarder_size) if self._is_train else self._boarder_size / 2
            y = nr.randint(self._boarder_size) if self._is_train else self._boarder_size / 2
            if nr.random() > 0.5 and self._is_train:
                trans_image = im[:, y:y+self._crop_size, x+self._crop_size:x:-1, :]
            else:
                trans_image = im[:, y:y+self._crop_size, x:x+self._crop_size, :]
            processed.append(trans_image.reshape(1, self._crop_size, self._crop_size, self._num_channels))
        # data = np.concatenate(processed, axis=0)
        data = np.reshape(processed, (len(imlist), self._crop_size, self._crop_size, self._num_channels))
        return data


    def transform_all_with_bbox(self, imlist, bbox, flag='anc'):
        # print 'transform'
        processed = []
        for id, im in enumerate(imlist):
            if im.shape[-1] == self._crop_size:
                processed.append(im.reshape(1, self._crop_size, self._crop_size, self._num_channels+1))
                continue

            def boundary_check(bbox):
                x, y, w, h = bbox
                x = min(max(0, x), 250)
                y = min(max(0, y), 250)
                w = min(max(5, w), 256 - x)
                h = min(max(5, h), 256 - y)
                return [x, y, w, h]

            def expand_boxes(bbox, ratio=1.2):
                x, y, w, h = bbox
                x_cen = x + 0.5 * w
                y_cen = y + 0.5 * h
                w *= ratio
                h *= ratio
                x = x_cen - w * 0.5
                y = y_cen - h * 0.5
                return np.array([x, y, w, h])

            # first create another blank image with the same size as the original image
            new_im = np.zeros(im.shape)
            if flag == 'anc':
                bb_part1 = bbox[id][0][0:4]
                bb_part2 = bbox[id][0][12:16]
            elif flag == 'pos':
                bb_part1 = bbox[id][0][4:8]
                bb_part2 = bbox[id][0][16:20]
            else:
                bb_part1 = bbox[id][0][8:12]
                bb_part2 = bbox[id][0][20:24]

            bbox1 = expand_boxes(bb_part1, ratio=1.2)
            bbox2 = expand_boxes(bb_part2, ratio=1.2)
            bb1 = np.round(bbox1).astype(np.int32).tolist()
            bb2 = np.round(bbox2).astype(np.int32).tolist()
            # print(bb)
            x1, y1, width1, height1 = boundary_check(bb1)
            x2, y2, width2, height2 = boundary_check(bb2)
            new_im[y1:y1+height1, x1:x1+width1, :] = 1.
            new_im[y2:y2+height2, x2:x2+width2, :] = 1.
            im = np.dstack((im, new_im))

            # translation and flip for image
            im = im.reshape(1, self._in_size, self._in_size, self._num_channels+1)
            x = nr.randint(self._boarder_size) if self._is_train else self._boarder_size / 2
            y = nr.randint(self._boarder_size) if self._is_train else self._boarder_size / 2
            if nr.random() > 0.5 and self._is_train:
                trans_image = im[:, y:y+self._crop_size, x+self._crop_size:x:-1, :]
            else:
                trans_image = im[:, y:y+self._crop_size, x:x+self._crop_size, :]
            processed.append(trans_image.reshape(1, self._crop_size, self._crop_size, self._num_channels+1))
        # data = np.concatenate(processed, axis=0)
        data = np.reshape(processed, (len(imlist), self._crop_size, self._crop_size, self._num_channels+1))
        return data


    def transform_all_part(self, imlist, bbox, flag='anc'):
        processed = []
        # get the bbox for anc/pos/neg
        if len(imlist)!=len(bbox):
            print 'The number of triplets is different with the number of bbox.'
            return
        bb_part1 = []
        bb_part2 = []
        if flag == 'anc':
            for i in range(len(bbox)):
                bb_part1.append(bbox[i][0][0:4])
                bb_part2.append(bbox[i][0][12:16])
        elif flag == 'pos':
            for i in range(len(bbox)):
                bb_part1.append(bbox[i][0][4:8])
                bb_part2.append(bbox[i][0][16:20])
        else:
            for i in range(len(bbox)):
                bb_part1.append(bbox[i][0][8:12])
                bb_part2.append(bbox[i][0][20:24])

        processed1=[]
        processed2=[]

        def boundary_check(bbox):
            x, y, w, h = bbox
            x = min(max(0, x), 250)
            y = min(max(0, y), 250)
            w = min(max(5, w), 256-x)
            h = min(max(5, h), 256-y)
            return [x, y, w, h]

        def expand_boxes(bbox, ratio=1.2):
            x, y, w, h = bbox
            x_cen = x + 0.5 * w
            y_cen = y + 0.5 * h
            w *= ratio
            h *= ratio
            x = x_cen - w * 0.5
            y = y_cen - h * 0.5
            return np.array([x, y, w, h])

        for id, im in enumerate(imlist):
            # if im.shape[-1] == self._crop_size:
            #     processed.append(im.reshape(1, self._crop_size, self._crop_size, self._num_channels))
            #     continue
            # crop image based on bounding box and then scale
            for i in range(2):
                if i == 0:
                    bb_part = bb_part1
                else:
                    bb_part = bb_part2
                # if len(bb_part[id])<4:
                #     bb_part[id] = [1,1,self._in_size-1,self._in_size-1]
                bbox = expand_boxes(bb_part[id])
                bb = np.round(bbox).astype(np.int32).tolist()
                # print(bb)
                x, y, width, height = boundary_check(bb)
                # x = int(bb_part[id][0])
                # y = int(bb_part[id][1])
                # width = int(bb_part[id][2])
                # height = int(bb_part[id][3])
                part_image = np.require(np.squeeze(im[y:y+height, x:x+width, :], axis=2), requirements='C')
                # print(part_image.shape)
                part_image = resize(part_image, [self._crop_size, self._crop_size], preserve_range=True)
                part_image = part_image[:, :, np.newaxis]
                # plt.imshow(part_image)
                # plt.show()
                if i == 0:
                    processed1.append(part_image.reshape(1, self._crop_size, self._crop_size, self._num_channels))
                else:
                    processed2.append(part_image.reshape(1, self._crop_size, self._crop_size, self._num_channels))

        # data = np.concatenate(processed, axis=0)
        data_part1 = np.reshape(processed1, (len(imlist), self._crop_size, self._crop_size, self._num_channels))
        data_part2 = np.reshape(processed2, (len(imlist), self._crop_size, self._crop_size, self._num_channels))
        return data_part1, data_part2


    def transform_bbox(self, imlist, bbox, flag='anc'):
        processed = []
        # get the bbox for anc/pos/neg
        if len(imlist)!=len(bbox):
            print 'The number of triplets is different with the number of bbox.'
            return
        bb_part1 = []
        bb_part2 = []
        if flag == 'anc':
            for i in range(len(bbox)):
                bb_part1.append(bbox[i][0][0:4])
                bb_part2.append(bbox[i][0][12:16])
        elif flag == 'pos':
            for i in range(len(bbox)):
                bb_part1.append(bbox[i][0][4:8])
                bb_part2.append(bbox[i][0][16:20])
        else:
            for i in range(len(bbox)):
                bb_part1.append(bbox[i][0][8:12])
                bb_part2.append(bbox[i][0][20:24])

        processed1=[]
        processed2=[]

        def boundary_check(bbox):
            x, y, w, h = bbox
            x = min(max(0, x), 250)
            y = min(max(0, y), 250)
            w = min(max(5, w), 256-x)
            h = min(max(5, h), 256-y)
            return [x, y, w, h]

        def expand_boxes(bbox, ratio=1.2):
            x, y, w, h = bbox
            x_cen = x + 0.5 * w
            y_cen = y + 0.5 * h
            w *= ratio
            h *= ratio
            x = x_cen - w * 0.5
            y = y_cen - h * 0.5
            return np.array([x, y, w, h])

        for id, im in enumerate(imlist):
            # if im.shape[-1] == self._crop_size:
            #     processed.append(im.reshape(1, self._crop_size, self._crop_size, self._num_channels))
            #     continue
            # crop image based on bounding box and then scale
            for i in range(2):
                if i == 0:
                    bb_part = bb_part1
                else:
                    bb_part = bb_part2
                # if len(bb_part[id])<4:
                #     bb_part[id] = [1,1,self._in_size-1,self._in_size-1]
                bbox = expand_boxes(bb_part[id])
                bb = np.round(bbox).astype(np.int32).tolist()
                # print(bb)
                x, y, width, height = boundary_check(bb)
                # x = int(bb_part[id][0])
                # y = int(bb_part[id][1])
                # width = int(bb_part[id][2])
                # height = int(bb_part[id][3])
                part_image = np.require(np.squeeze(im[y:y+height, x:x+width, :], axis=2), requirements='C')
                # print(part_image.shape)
                part_image = resize(part_image, [self._crop_size, self._crop_size], preserve_range=True)
                part_image = part_image[:, :, np.newaxis]
                # plt.imshow(part_image)
                # plt.show()
                if i == 0:
                    processed1.append(part_image.reshape(1, self._crop_size, self._crop_size, self._num_channels))
                else:
                    processed2.append(part_image.reshape(1, self._crop_size, self._crop_size, self._num_channels))

        # data = np.concatenate(processed, axis=0)
        data_part1 = np.reshape(processed1, (len(imlist), self._crop_size, self._crop_size, self._num_channels))
        data_part2 = np.reshape(processed2, (len(imlist), self._crop_size, self._crop_size, self._num_channels))
        return data_part1, data_part2