import numpy as np
from PIL import Image
import os
label_background = 5            # Set background label as 255. There are 5 meaningful classes. labeling 0, 1, 2, 3, 4
im_w = 512
im_h = 512


image_dir_name = 'Images'
label_dir_name = 'labelsAsNpy'

# inputdir = 'C:\Users\gdai\Downloads\Training Datasets'
inputdir = '/home/gxdai/Guoxian_Dai/data/medicalImage/wustl/TrainingSet'
labels = ['Esophagus', 'Heart', 'Lung_L', 'Lung_R', 'SpinalCord']
samplelist = os.listdir(inputdir)       # Get the sample list
for sample in samplelist:
    print sample
    datapath = os.path.join(inputdir, sample, image_dir_name)     # data path for each sample
    labeldir = os.path.join(inputdir, sample, label_dir_name)
    if not os.path.isdir(labeldir):
        os.mkdir(labeldir)
    slidelist = os.listdir(datapath)                        # The slides list for each sample
    for slide in slidelist:                                 # Get the label of slide
        print(slide)
        labelM = np.full((im_w, im_h), label_background)                       # initialize labelM with all zeros
        for index in range(5):
            label = labels[index]
            labelpath = os.path.join(inputdir, sample, 'Masks', label)
            print("This is the %d-th labelpath" %(index))
            if not os.path.isdir(labelpath):
                continue
            labellist = os.listdir(labelpath)       # all the labeled files in the current folder
            label_check = slide[:-4]+'_mask'+'.tif'
            for tmp in labellist:
                if label_check == tmp:
                    im = Image.open(os.path.join(labelpath, tmp))
                    im_data = list(im.getdata())
                    im_data = np.array(im_data)
                    im_data = np.reshape(im_data, (im_w, im_h))
                    location = np.where(im_data == 255)     # get the label location
                    labelM[location] = index               #
                    # TO be continued
                    #print("min(im_data) = {}".format(np.amin(im_data[:])))
                    #print("max(im_data) = {}".format(np.amax(im_data[:])))
                    #print(im_data.shape)

        labelfile = os.path.join(labeldir, slide[:-4] + '.npy')             # The output file name
        np.save(labelfile, labelM)
        # print("min(labelM) = {}".format(np.amin(labelM)))
        # print("max(labelM) = {}".format(np.amax(labelM)))
        # im_label = Image.fromarray(labelM.astype(np.uint8))
        # im_label.save(labelfile)
        # show_label = Image.open(labelfile)
        # show_label = np.array(list(show_label.getdata()))
        # print("min(show_label) = {}".format(np.amin(show_label)))
        # print("max(show_label) = {}".format(np.amax(show_label)))

