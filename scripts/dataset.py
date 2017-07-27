import os
import numpy as np
from PIL import Image
import argparse
import time


class Dataset(object):
    def __init__(self, traindir, testdir, batchsize=6, seqLength=15, step_for_batch=2, step_for_update=5, class_num = 6):
        """
        :param traindir: string
        :param testdir: string
        :param batchsize: string
        """
        self.traindir = traindir                        # get train dir
        self.testdir = testdir                          # get test dir
        self.batchsize = batchsize                      # get batch size
        self.class_num = class_num
        self.image_dir_name = 'Images'
        self.label_dir_name = "labelsAsNpy"

        self.trainSampleList = self.getfilelist(self.traindir)       # get the train sample list
        self.trainSampleNum = len(self.trainSampleList)
        self.testSampleList = self.getfilelist(self.testdir)         # get the test sample list
        self.testSampleNum = len(self.testSampleList)
        self.trainSlideDataDict, self.trainSlideDataDictNum, self.trainSlideLabelDict, self.trainSlideDataDictNum = self.genslidelist()
        self.seqLength = seqLength
        self.step_for_batch = step_for_batch
        self.total_step = (self.batchsize-1)* self.step_for_batch + self.seqLength                           # every "step" pick one sequence
        self.step_for_update = step_for_update                                           # every certain of examples, get to the next examples.



        self.train_ptr = 0                          # training sample pointer
        self.train_sub_ptr = 0                      # training slide pointer
        self.test_ptr = 0                           # test sample pointer
        self.test_sub_ptr = 0                       # test slides pointer
    def getfilelist(self, root_dir):
        return sorted(os.listdir(root_dir))         # sort the returned list

    def genslidelist(self):
        # For all the training samples
        trainSlideDataDict = {}         # save all the training slides for each sample, key is sample name
        trainSlideDataDictNum = {}      # save the number of training slides for each sample, key is smaple name
        trainSlideLabelDict = {}         # save all the training slides for each sample, key is sample name
        trainSlideLabelDictNum = {}      # save the number of training slides for each sample, key is smaple name
        for tmpdir in self.trainSampleList:
            # get data
            tmppath = os.path.join(self.traindir, tmpdir, self.image_dir_name)
            tmplist = sorted(os.listdir(tmppath))           # This sorted doesn't work
            trainSlideDataDict[tmpdir] = tmplist
            trainSlideDataDictNum[tmpdir] = len(tmplist)
            # Reorder the list in an increasing order
            ext = tmplist[0].split('.')[-1]
            trainSlideDataDict[tmpdir] = [tmpdir + '_' + str(i) + '.' + ext for i in range(1, trainSlideDataDictNum[tmpdir]+1, 1)]     # Resort the  file in an increasing order
            # get label
            tmppath = os.path.join(self.traindir, tmpdir, self.label_dir_name)
            tmplist = sorted(os.listdir(tmppath))
            trainSlideLabelDict[tmpdir] = tmplist
            trainSlideLabelDictNum[tmpdir] = len(tmplist)
            # Reorder the list in an increasing order
            ext = tmplist[0].split('.')[-1]
            trainSlideLabelDict[tmpdir] = [tmpdir + '_' + str(i) + '.' + ext for i in range(1, trainSlideLabelDictNum[tmpdir]+1, 1)]     # Resort the  file in an increasing order
            if trainSlideDataDictNum[tmpdir] != trainSlideLabelDictNum[tmpdir]:
                print("The totoal number of data and label are different******")
        return trainSlideDataDict, trainSlideDataDictNum, trainSlideLabelDict, trainSlideDataDictNum


    def next_batch(self, phase):
        start_time = time.time()
        if phase == 'train':
            self.train_ptr = self.train_ptr % self.trainSampleNum
            if self.train_sub_ptr + self.total_step <= self.trainSlideDataDictNum[self.trainSampleList[self.train_ptr]]:
                sample = self.trainSampleList[self.train_ptr]
                datapath = [os.path.join(self.traindir, sample, self.image_dir_name, self.trainSlideDataDict[sample][self.train_sub_ptr+i*self.step_for_batch+j]) \
                            for i in range(self.batchsize) for j in range(self.seqLength)]
                labelpath = [os.path.join(self.traindir, sample, self.label_dir_name, self.trainSlideLabelDict[sample][self.train_sub_ptr+i*self.step_for_batch+j]) \
                             for i in range(self.batchsize) for j in range(self.seqLength)]
                self.train_sub_ptr += self.step_for_update
            elif self.total_step <= self.trainSlideDataDictNum[self.trainSampleList[self.train_ptr]]:        # Pick the last batch
                sample = self.trainSampleList[self.train_ptr]
                self.train_sub_ptr = self.trainSlideDataDictNum[self.trainSampleList[self.train_ptr]] - self.total_step
                # update sub pointer
                datapath = [os.path.join(self.traindir, sample, self.image_dir_name, self.trainSlideDataDict[sample][self.train_sub_ptr+i*self.step_for_batch+j]) \
                            for i in range(self.batchsize) for j in range(self.seqLength)]
                labelpath = [os.path.join(self.traindir, sample, self.label_dir_name, self.trainSlideLabelDict[sample][self.train_sub_ptr+i*self.step_for_batch+j]) \
                             for i in range(self.batchsize) for j in range(self.seqLength)]
                self.train_ptr += 1                 # go to next sample
                self.train_ptr = self.train_ptr % self.trainSampleNum
                self.train_sub_ptr = 0
        elif phase == 'test':
            self.test_ptr = self.test_ptr % self.testSampleNum
            if self.test_sub_ptr + self.total_step < self.testSlideDataDictNum[self.testSampleList[self.test_ptr]]:
                sample = self.testSampleList[self.test_ptr]
                datapath = [os.path.join(self.testdir, sample, self.image_dir_name, self.testSlideDataDict[sample][self.test_sub_ptr+i*self.step_for_batch+j]) \
                            for i in range(self.batchsize) for j in range(self.seqLength)]
                labelpath = [os.path.join(self.testdir, sample, self.label_dir_name, self.testSlideLabelDict[sample][self.test_sub_ptr+i*self.step_for_batch+j]) \
                             for i in range(self.batchsize) for j in range(self.seqLength)]
                self.test_sub_ptr += self.step_for_update
            else:
                self.test_ptr += 1                 # go to next sample
                self.test_ptr = self.test_ptr % self.testSampleNum
                self.test_sub_ptr = 0              # Reset starting pointer
                if self.test_sub_ptr + self.total_step < self.testSlideDataDictNum[self.testSampleList[self.test_ptr]]:
                    sample = self.testSampleList[self.test_ptr]
                    datapath = [os.path.join(self.testdir, sample, self.image_dir_name, self.testSlideDataDict[sample][self.test_sub_ptr+i*self.step+j]) \
                            for i in range(self.batchsize) for j in range(self.seqLength)]
                    labelpath = [os.path.join(self.testdir, sample, self.label_dir_name, self.testSlideLabelDict[sample][self.test_sub_ptr+i*self.step+j]) \
                             for i in range(self.batchsize) for j in range(self.seqLength)]
                    self.test_sub_ptr += self.step_for_update
        print("GET the data path takes {} seconds ".format(time.time() - start_time))
        im_w = 512
        im_h = 512
        im_c = 1
        img_shape = (im_w, im_h, im_c)
        img_array = np.zeros((self.batchsize*self.seqLength, im_w, im_h, im_c))
        label_array = np.zeros((self.batchsize*self.seqLength, im_w, im_h, self.class_num))
        start_time = time.time()
        for i in range(self.batchsize*self.seqLength):
            img_array[i] = self.load_img(datapath[i], img_shape)
            # This is for debug
            # tmp_check_label = self.load_label_npy(labelpath[i], img_shape)
            # tmp_check_label = np.argmax(tmp_check_label, axis=2)
            # shape = tmp_check_label.shape
            # tmp_check_label = np.reshape(tmp_check_label, shape[0]*shape[1])
            # print(datapath[i])
            # for index_j in range(6):
            #     location = np.where(tmp_check_label == index_j)
            #     num = location[0].shape[0]
            #     print("The number of points for label {}:\t {}".format(index_j, num))
            label_array[i] = self.load_label_npy(labelpath[i], img_shape)
        print("Loading data takes {} seconds".format(time.time() - start_time))
        return img_array, label_array
    def load_img(self, path, shape):
        img = Image.open(path)
        img = np.array(list(img.getdata()))
        img = np.reshape(img, shape).astype(np.uint8)
        return img
    def load_label(self, path, shape):
        img = Image.open(path)
        img = np.array(list(img.getdata()))
        # label
        label_array = np.zeros((shape[0]*shape[1], self.class_num))
        for i in range(shape[0]*shape[1]):
            if img[i] > 5:
                print("img[i]={}".format(img[i]))
                continue
            label_array[i, img[i]] = 1
        label_array = np.reshape(label_array, (shape[0], shape[1], self.class_num)).astype(np.uint8)
        return label_array
    def load_label_npy(self, path, shape):
        img = np.load(path)
        img = np.reshape(img, (shape[0]*shape[1]))
        print("max(img) = {}".format(np.amax(img)))
        print("min(img) = {}".format(np.amin(img)))
        print(shape[0]*shape[1])
        print(img.shape)
        # label
        label_array = np.zeros((shape[0]*shape[1], self.class_num))
        # for index_i in range(shape[0]):
        #     for index_j in range(shape[1]):
        #         label_array[index_i, index_j, img[index_i, index_j]] = 1
        for index in range(shape[0]*shape[1]):
            label_array[index, img[index]] = 1
        label_array = np.reshape(label_array, (shape[0], shape[1], self.class_num)).astype(np.uint8)
        return label_array

    def checkList(self, phase):
        if phase == 'train':
            self.train_ptr = self.train_ptr % self.trainSampleNum
            if self.train_sub_ptr + self.total_step <= self.trainSlideDataDictNum[self.trainSampleList[self.train_ptr]]:
                sample = self.trainSampleList[self.train_ptr]
                datapath = [os.path.join(self.traindir, sample, self.image_dir_name, self.trainSlideDataDict[sample][self.train_sub_ptr+i*self.step_for_batch+j]) \
                            for i in range(self.batchsize) for j in range(self.seqLength)]
                labelpath = [os.path.join(self.traindir, sample, self.label_dir_name, self.trainSlideLabelDict[sample][self.train_sub_ptr+i*self.step_for_batch+j]) \
                             for i in range(self.batchsize) for j in range(self.seqLength)]
                self.train_sub_ptr += self.step_for_update
            elif self.total_step <= self.trainSlideDataDictNum[self.trainSampleList[self.train_ptr]]:        # Pick the last batch
                sample = self.trainSampleList[self.train_ptr]
                self.train_sub_ptr = self.trainSlideDataDictNum[self.trainSampleList[self.train_ptr]] - self.total_step
                # update sub pointer
                datapath = [os.path.join(self.traindir, sample, self.image_dir_name, self.trainSlideDataDict[sample][self.train_sub_ptr+i*self.step_for_batch+j]) \
                            for i in range(self.batchsize) for j in range(self.seqLength)]
                labelpath = [os.path.join(self.traindir, sample, self.label_dir_name, self.trainSlideLabelDict[sample][self.train_sub_ptr+i*self.step_for_batch+j]) \
                             for i in range(self.batchsize) for j in range(self.seqLength)]
                self.train_ptr += 1                 # go to next sample
                self.train_ptr = self.train_ptr % self.trainSampleNum
                self.train_sub_ptr = 0
            # else:
            #     self.train_ptr += 1                 # go to next sample
            #     self.train_ptr = self.train_ptr % self.trainSampleNum
            #     self.train_sub_ptr = 0              # Reset starting pointer
            #     if self.train_sub_ptr + self.total_step - 1 <= self.trainSlideDataDictNum[self.trainSampleList[self.train_ptr]]:
            #         sample = self.trainSampleList[self.train_ptr]
            #         datapath = [os.path.join(self.traindir, sample, self.image_dir_name, self.trainSlideDataDict[sample][self.train_sub_ptr+i*self.step_for_batch+j]) \
            #                 for i in range(self.batchsize) for j in range(self.seqLength)]
            #         labelpath = [os.path.join(self.traindir, sample, self.label_dir_name, self.trainSlideLabelDict[sample][self.train_sub_ptr+i*self.step_for_batch+j]) \
            #                  for i in range(self.batchsize) for j in range(self.seqLength)]
            #         self.train_sub_ptr += self.step_for_update
        elif phase == 'test':
            self.test_ptr = self.test_ptr % self.testSampleNum
            if self.test_sub_ptr + self.total_step < self.testSlideDataDictNum[self.testSampleList[self.test_ptr]]:
                sample = self.testSampleList[self.test_ptr]
                datapath = [os.path.join(self.testdir, sample, self.image_dir_name, self.testSlideDataDict[sample][self.test_sub_ptr+i*self.step_for_batch+j]) \
                            for i in range(self.batchsize) for j in range(self.seqLength)]
                labelpath = [os.path.join(self.testdir, sample, self.label_dir_name, self.testSlideLabelDict[sample][self.test_sub_ptr+i*self.step_for_batch+j]) \
                             for i in range(self.batchsize) for j in range(self.seqLength)]
                self.test_sub_ptr += self.step_for_update
            else:
                self.test_ptr += 1                 # go to next sample
                self.test_ptr = self.test_ptr % self.testSampleNum
                self.test_sub_ptr = 0              # Reset starting pointer
                if self.test_sub_ptr + self.total_step < self.testSlideDataDictNum[self.testSampleList[self.test_ptr]]:
                    sample = self.testSampleList[self.test_ptr]
                    datapath = [os.path.join(self.testdir, sample, self.image_dir_name, self.testSlideDataDict[sample][self.test_sub_ptr+i*self.step+j]) \
                            for i in range(self.batchsize) for j in range(self.seqLength)]
                    labelpath = [os.path.join(self.testdir, sample, self.label_dir_name, self.testSlideLabelDict[sample][self.test_sub_ptr+i*self.step+j]) \
                             for i in range(self.batchsize) for j in range(self.seqLength)]
                    self.test_sub_ptr += self.step_for_update
        return datapath, labelpath



