import tensorflow as tf
from scripts.dataset import Dataset
import argparse


parser = argparse.ArgumentParser(description='Process some directory.')
# traindir, testdir, batchsize=6, seqLength=15, step_for_batch=2, step_for_update=5):
parser.add_argument("--traindir", type=str)
parser.add_argument("--testdir", type=str)
parser.add_argument("--batchsize", type=int)
parser.add_argument("--seqLength", type=int)
parser.add_argument("--step_for_batch", type=int)
parser.add_argument("--step_for_update", type=int)
parser.add_argument("--class_num", type=int)

args = parser.parse_args(["--traindir=/home/gxdai/Guoxian_Dai/data/medicalImage/wustl/TrainingSet", "--testdir=/home/gxdai/Guoxian_Dai/data/medicalImage/wustl/TrainingSet", "--batchsize=3", "--seqLength=5", "--step_for_batch=5", "--step_for_update=15", "--class_num=6"])
data_provider = Dataset(args.traindir, args.testdir, args.batchsize, args.seqLength, args.step_for_batch, args.step_for_update)


# from tf_unet_convlstm import unet, util, image_util
from tf_unet_convlstm import unet_convlstm, util, image_util
import numpy as np
# net = unet.Unet(layers=3, features_root=64, channels=1, n_class=2)
net = unet_convlstm.Unet(layers=3, features_root=16, channels=1, n_class=6)

init = tf.global_variables_initializer()
trainer = unet_convlstm.Trainer(net)
output_path = './result_random'
path = trainer.train(data_provider, output_path, training_iters=1000, epochs=50, restore = True, mode='test')
"""
with tf.Session() as sess:
	sess.run(init)
	logits = sess.run(net.logits, feed_dict={net.x: images, net.keep_prob: 0.1})
	print(logits.shape)

"""

