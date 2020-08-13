#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import caffe
import numpy
import glob
import os.path
import sys

if __name__=='__main__':
  if not os.path.exists('images'):
    print('Place images to be classified in images/brick/*.jpg, images/carpet/*.jpg, ...')
    sys.exit(1)
  categories=[x.strip() for x in open('categories.txt').readlines()]

  arch='vgg16' # googlenet, vgg16 or alexnet
  net=caffe.Classifier('remove-fc-{}.prototxt'.format(arch),'minc-{}.caffemodel'.format(arch),channel_swap=(2,1,0),mean=numpy.array([104,117,124]))
  net.save('remove-fc-{}.caffemodel'.format(arch))
  # result={}
  # for i,x in enumerate(categories):
  #   result[x]=[]
  #   for j,y in enumerate(sorted(glob.glob('images/{}/*'.format(x)))):
  #     z=net.predict([caffe.io.load_image(y)*255.0])[0]
  #     k=z.argmax()
  #     print(arch,y,categories[k],z[k],k==i)
  #     result[x].append(k==i)
  # for i,x in enumerate(categories):
  #   print(arch,x,sum(result[x])/len(result[x]))
  # print(arch,sum(sum(x) for x in result.values())/sum(len(x) for x in result.values()))