import os 
import caffe
import numpy as np

root='/home/z/lunaexp/'
deploy=root+'VGG/64/deploy_dp45.prototxt'
caffe_model=root+'VGG/VGG16-64-27gray-dp45_2_iter_30000.caffemodel'

mean_file=root+'image/27view_gray/224/mean_train.npy'

caffe.set_mode_gpu()
net=caffe.Net(deploy,caffe_model,caffe.TEST)

#for image-processing
transformer=caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data',(2,0,1))#change(64,64,3)to(3,64,64)
transformer.set_mean('data',np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data',255)
transformer.set_channel_swap('data',(2,1,0))#RGB to BGR

#test images
import glob
test_img_dir=root+'cube/KA/KA_3image/'
test_result_dir=root+'VGG/KA_3image_res.txt'

#imgsextension=['jpg']
#os.chdir(test_img_dir)
#imglist=[]
#namelist=[]
#for extension in imgsextension:
    #extension='*.'+extension
    #imglist.append([os.path.realpath(e) for e in glob.glob(extension)])
    #namelist.append([e for e in glob.glob(extension)])

f=open(test_result_dir,'w')
for imgname in os.listdir(test_img_dir):
    img=(test_img_dir+imgname)
    im=caffe.io.load_image(img)
    net.blobs['data'].data[...]=transformer.preprocess('data',im)
    out=net.forward()
    prob=net.blobs['prob'].data[0].flatten()
    index=prob.argsort()[-1]
    f.write(imgname+' '+str(prob[0])+' '+str(prob[1])+' '+str(index)+'\n')
f.close
