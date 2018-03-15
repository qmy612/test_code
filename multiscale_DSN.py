import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm

import scipy.misc
from PIL import Image
import scipy.io
import os
from timeit import default_timer as timer



# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples/hed/
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

data_root = '';
test_lst = []
with open(data_root+'test_for_2016.lst') as f:
    test_lst = f.readlines()
    
test_lst = [data_root+x.strip() for x in test_lst]

#remove the following two lines if testing with cpu
caffe.set_mode_gpu()
caffe.set_device(0)
start = timer()

# load net
model_root = './'

for idx in range(0, len(test_lst)):
    im = Image.open(test_lst[idx]).convert('L')
    in_ = np.array(im, dtype=np.float32)
    ret = np.empty((in_.shape[0], in_.shape[1], 3), dtype=np.float32)
    ret[:, :, 0] = in_
    ret[:, :, 1] = in_
    ret[:, :, 2] = in_
    max_V = in_.max()
    min_V = in_.min()
    mean_V = (max_V + min_V)/2
    ret -= np.array((mean_V, mean_V, mean_V))

    print ret.shape

    in_ = ret

    im_height = in_.shape[0]
    im_width = in_.shape[1]
    min_D = min(im_height, im_width)
    max_D = max(im_height, im_width)

    win_sz = 0
    if (max_D/min_D >= 4):
       win_sz = min_D
    else:
       win_sz = int(min_D/2)

    # upscale or downscale
    scale = 1
    if (win_sz > 410):
       scale = 410 / float(win_sz)
       win_sz = 410

    if (win_sz < 192):
       scale = 192 / float(win_sz)
       win_sz = 192

    print scale

    if (scale != 1):
       in_ = scipy.ndimage.zoom(in_[:,:,0], scale, order=1)
       ret = np.empty((in_.shape[0], in_.shape[1], 3), dtype=np.float32)
       ret[:, :, 0] = in_
       ret[:, :, 1] = in_
       ret[:, :, 2] = in_
       in_ = ret

    print in_.shape
    print max_D, min_D, win_sz

    height = in_.shape[0]
    width = in_.shape[1]
    output_conv3 = np.zeros((in_.shape[0], in_.shape[1]), dtype=np.float32)
    output_conv4 = np.zeros((in_.shape[0], in_.shape[1]), dtype=np.float32)
    output_conv5 = np.zeros((in_.shape[0], in_.shape[1]), dtype=np.float32)

    for it in range(1, 4):
	print it
	if it == 1:
    	   net = caffe.Net(model_root+'conv_3/deploy_conv3.prototxt', model_root+'conv_3/conv_3.caffemodel', caffe.TEST)
	if it == 2:
    	   net = caffe.Net(model_root+'conv_4/deploy_conv4.prototxt', model_root+'conv_4/conv_4.caffemodel', caffe.TEST)
	if it == 3:
    	   net = caffe.Net(model_root+'conv_5/deploy_conv5.prototxt', model_root+'conv_5/conv_5.caffemodel', caffe.TEST)

    	for r in range(0, height, int(win_sz/2)):
	    for c in range(0, width, int(win_sz/2)):
	    	b_b = r+win_sz
	   	r_b = c+win_sz

	   	if (b_b >= height):
	   	    b_b = height - 1
	   	    r = b_b - win_sz + 1

	   	if (r_b >= width):
	   	    r_b = width - 1
	   	    c = r_b - win_sz + 1
	    
	    	print r, c, b_b, r_b, win_sz

	   	patch = in_[r:b_b, c:r_b,:]	

	   	print patch.shape

	    	patch = patch.transpose((2,0,1))

		# shape for C5 input (data blob is N x C x H x W),
	        net.blobs['data'].reshape(1, *patch.shape)
	        net.blobs['data'].data[...] = patch
	        net.forward()
		fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]

		if it == 1:
		   tmp = output_conv3[r:b_b, c:r_b]
		   tmp = np.maximum(tmp, fuse)
		   output_conv3[r:b_b, c:r_b] = tmp
		if it == 2:
		   tmp = output_conv4[r:b_b, c:r_b]
		   tmp = np.maximum(tmp, fuse)
		   output_conv4[r:b_b, c:r_b] = tmp
		if it == 3:
		   tmp = output_conv5[r:b_b, c:r_b]
		   tmp = np.maximum(tmp, fuse)
		   output_conv5[r:b_b, c:r_b] = tmp
 
    multi_scale = []
    multi_scale.append(output_conv3)
    multi_scale.append(output_conv4)
    multi_scale.append(output_conv5)
    multi_scale = np.amin(multi_scale, axis = 0)
    #multi_scale = scipy.misc.imresize(multi_scale, (im_height, im_width), interp='bicubic', mode='L')
    multi_scale = scipy.ndimage.zoom(multi_scale, 1/scale, order=1)
    output = np.zeros((im_height, im_width), dtype=np.float32)
    output[0:np.minimum(im_height, multi_scale.shape[0])-1, 0:np.minimum(im_width, multi_scale.shape[1])-1] = multi_scale[0:np.minimum(im_height, multi_scale.shape[0])-1, 0:np.minimum(im_width, multi_scale.shape[1])-1]
    multi_scale = output
    thresh = 0.84*multi_scale.max();
    multi_scale[multi_scale <= thresh] = 0;
    multi_scale[multi_scale > thresh] = 255;
    multi_scale = 255 - multi_scale;

    pos = [i for i, ltr in enumerate(test_lst[idx]) if ltr == '/']
    print test_lst[idx][pos[-1]+1:]
    scipy.misc.imsave(data_root + 'IDCard_Binarization_output/' + test_lst[idx][pos[-1]+1:], multi_scale)
	
end = timer()
print (end - start)/16
   
