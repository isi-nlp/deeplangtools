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
  parser = argparse.ArgumentParser(description="Do gradient descent for the n matrix interlingus embedding experiment",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--edim", "-e", default=128, type=int, help="dimension of embeddings")
  parser.add_argument("--idim", "-i", default=128, type=int, help="dimension of interlingus")
  parser.add_argument("--langs", "-l", default=3, type=int, help="number of languages to generate data for")
  parser.add_argument("--lang1file", "-1", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="side 1 input file (assume word lang vec)")
  parser.add_argument("--lang2file", "-2", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="side 2 input file (assume word lang vec)")
  # TODO: base vocabulary for more realistic dev statistics
  parser.add_argument("--devsize", "-s", type=int, default=0, help="How much dev data to sample; rest is training")
  parser.add_argument("--modelfile", "-f", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="all models output file")
  parser.add_argument("--learningrate", "-r", type=float, default=0.001, help="Learning rate")
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
  mats = []
  paramrange=args.parammax-args.parammin
  for i in xrange(args.langs):
    mats.append(np.matrix((np.random.rand(args.edim, args.idim)*paramrange)+args.parammin))

  matshape = (args.edim, args.idim)
  # load in data. form monolingual vocabularies for later evaluation
  # input format is word (string), language (int), vector (floats)
  strain_in = np.matrix(np.loadtxt(args.lang1file, dtype=str))
  strain_out = np.matrix(np.loadtxt(args.lang2file, dtype=str))
  vocab = dd(lambda: dict())
  for entry in np.concatenate((strain_in, strain_out), axis=0):
    word = entry[0,0]
    lang = entry[0,1]
    vec = entry[0,2:].astype(float)
    if word not in vocab[lang]:
      vocab[lang][word]=vec
  # TODO: here is where you add external monolingual data
  # TODO: check for duplicate non-same entries

  # TODO: make nn indices for each language
  
  # now it's just numeric lang, vec
  strain_in = strain_in[:,1:].astype(float)
  strain_out = strain_out[:,1:].astype(float)
  print "data loaded"
  # move language indices to the beginning of the vector
  data = np.concatenate((strain_in[:,0], strain_out[:,0], strain_in[:,1:], strain_out[:,1:]), axis=1)
  np.random.shuffle(data)
  devdata = data[:args.devsize]
  data = data[args.devsize:]
  if len(devdata) == 0:
    devdata = data
  
  batchcount = data.shape[0]/args.minibatch # some data might be left but it's shuffled each time
  for iteration in xrange(args.iterations):
    np.random.shuffle(data)
    for batchnum in xrange(batchcount):
#      print "batch "+str(batchnum)
      rowstart=batchnum*args.minibatch
      rowend = rowstart+args.minibatch
      langs = data[rowstart:rowend, :2].astype(int)
      all_batch_in = data[rowstart:rowend, 2:args.edim+2]
      all_batch_out = data[rowstart:rowend, args.edim+2:]
      updates = [np.zeros(matshape) for x in xrange(args.langs)]
      counts = np.zeros(args.langs)
      # need to pick different matrices to update for each item so this will be slow
      for i in xrange(len(all_batch_in)):
        inlang=langs[i,0]
        outlang=langs[i,1]
        counts[inlang]+=1
        counts[outlang]+=1
        batch_in = all_batch_in[i]
        batch_out = all_batch_out[i]
        smat = mats[inlang]
        tmat = mats[outlang]
        im = batch_in*smat
        imn= im*(tmat.transpose())
        delta = imn-batch_out
        # learning
        twodel = 2*delta
#        print im.shape
#        print twodel.shape
#        print updates[outlang].shape
        updates[outlang] += (im.transpose()*twodel).transpose()
#        print batch_in.shape
#        print twodel.shape
#        print tmat.shape
#        print updates[inlang].shape
        updates[inlang] += batch_in.transpose()*twodel*tmat
      for l in xrange(args.langs):
        if counts[l] > 0:
          mats[l] = mats[l] - (updates[l]*args.learningrate/counts[l])
      # print l2nrm, the learned vector, and the normalized learned vector
    if iteration % args.period == 0: # report full training objective
      langs = devdata[:, :2].astype(int)
      all_batch_in = devdata[:, 2:args.edim+2]
      all_batch_out = devdata[:, args.edim+2:]
      l2n2 = 0
      allimn = []
      for i in xrange(len(all_batch_in)):
        inlang=langs[i,0]
        outlang=langs[i,1]
        batch_in = all_batch_in[i]
        batch_out = all_batch_out[i]
        smat = mats[inlang]
        tmat = mats[outlang]
        im = batch_in*smat
        imn= im*(tmat.transpose())
        allimn.append(imn)
        delta = imn-batch_out
        delnorm = LA.norm(delta, ord=2)
        l2n2 += delnorm*delnorm
        # TODO: find 0/1 nn accuracy and rank average
      allimn = np.vstack(allimn)
      print "\nIteration "+str(iteration)+" = "+str(l2n2) + " =\n" + str(allimn[:2,:10]) + "\nvs\n" + str(all_batch_out[:2,:10])
  matsasdict = dict([(str(x[0]), x[1]) for x in enumerate(mats)])
  np.savez_compressed(args.modelfile, **matsasdict)

if __name__ == '__main__':
  main()

