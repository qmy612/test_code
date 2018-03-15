

import caffe
import surgery,score


import numpy as np
import os

#import setproctitle
#setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = 'd80k500dp-train_iter_5000.caffemodel'

# init
#gpu&device
#caffe.set_device(int(sys.argv[1]))
#caffe.set_mode_gpu()
caffe.set_mode_cpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('E:/QMYdata/val.txt', dtype=str)

for _ in range(10):
    solver.step(1000)
    score.seg_tests(solver, False, val, layer='fc8_newk')
