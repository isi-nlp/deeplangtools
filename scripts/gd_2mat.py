#! /usr/bin/env python
import argparse
import sys
import codecs
from itertools import izip
from collections import defaultdict as dd
import re
import os.path
import numpy as np
from numpy import linalg as LA
scriptdir = os.path.dirname(os.path.abspath(__file__))

def main():
  parser = argparse.ArgumentParser(description="Do gradient descent for the two matrix interlingus embedding experiment",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--edim", "-e", default=128, type=int, help="dimension of embeddings")
  parser.add_argument("--idim", "-i", default=128, type=int, help="dimension of interlingus")
  parser.add_argument("--lang1file", "-1", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="L1 input file")
  parser.add_argument("--lang2file", "-2", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="L2 input file")
  parser.add_argument("--devsize", "-s", type=int, default=0, help="How much dev data to sample; rest is training")
  parser.add_argument("--modelafile", "-a", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="model a output file")
  parser.add_argument("--modelbfile", "-b", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="model b output file")
  parser.add_argument("--learningrate", "-l", type=float, default=0.001, help="Learning rate")
  parser.add_argument("--minibatch", "-m", type=int, default=10, help="Minibatch size")
  parser.add_argument("--iterations", "-t", type=int, default=100, help="Number of training iterations")
  parser.add_argument("--period", "-p", type=int, default=10, help="How often to look at objective")
  parser.add_argument("--parammin",  default=-2, type=float, help="minimum model parameter value")
  parser.add_argument("--parammax",  default=2, type=float, help="maximum model parameter value")



  try:
    args = parser.parse_args()
  except IOError, msg:
    parser.error(str(msg))

  # make initial transformation matrices
  paramrange=args.parammax-args.parammin
  smat = np.matrix((np.random.rand(args.edim, args.idim)*paramrange)+args.parammin)
  tmat = np.matrix((np.random.rand(args.edim, args.idim)*paramrange)+args.parammin).transpose()

  # load in data
  strain_in = np.matrix(np.loadtxt(args.lang1file))
  strain_out = np.matrix(np.loadtxt(args.lang2file))

  data = np.concatenate((strain_in, strain_out), axis=1)
  np.random.shuffle(data)
  devdata = data[:args.devsize]
  data = data[args.devsize:]

  batchcount = data.shape[0]/args.minibatch # some data might be left but it's shuffled each time
  for iteration in xrange(args.iterations):
    np.random.shuffle(data)
    for batchnum in xrange(batchcount):
      rowstart=batchnum*args.minibatch
      rowend = rowstart+args.minibatch
      batch_in = data[rowstart:rowend, :args.edim]
      batch_out = data[rowstart:rowend, args.edim:]
      im = batch_in*smat
      imn= im*tmat
      delta = imn-batch_out
      # learning
      twodel = 2*delta
      nupdate = im.transpose()*twodel/args.minibatch
      mupdate = batch_in.transpose()*twodel*tmat.transpose()/args.minibatch
      smat = smat-(mupdate*args.learningrate)
      tmat = tmat-(nupdate*args.learningrate)
      # print l2nrm, the learned vector, and the normalized learned vector
    if iteration % args.period == 0: # report full training objective
      batch_in = devdata[:, :args.edim]
      batch_out = devdata[:, args.edim:]
      im = batch_in*smat
      imn= im*tmat
      delta = imn-batch_out
      delnorm = LA.norm(delta, ord=2)
      l2n2 = delnorm*delnorm
      print str(l2n2) + " = " + str(imn[:2,:10]) + " vs " + str(batch_out[:2,:10])
  np.savetxt(args.modelafile, smat)
  np.savetxt(args.modelbfile, tmat)
  # TODO: save off matrices
if __name__ == '__main__':
  main()

