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
  parser = argparse.ArgumentParser(description="Do gradient descent for the n matrix interlingus embedding experiment with inverse-based updating",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--dim", "-d", default=128, type=int, help="dimension of embeddings and interlingus")
  parser.add_argument("--langs", "-l", default=3, type=int, help="number of languages to learn for")
  parser.add_argument("--dictionaries", "-c", nargs='+', type=argparse.FileType('r'), default=[sys.stdin,], help="vocabulary dictionaries of the form word lang vec")
  parser.add_argument("--infile", "-i", type=argparse.FileType('r'), default=sys.stdin, help="training instruction of the form word1 lang1 lang2 word2")
  parser.add_argument("--devsize", "-s", type=int, default=0, help="How much dev data to sample; rest is training")
  parser.add_argument("--modelfile", "-f", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="all models output file")
  parser.add_argument("--learningrate", "-r", type=float, default=0.001, help="Learning rate")
  parser.add_argument("--minibatch", "-m", type=int, default=10, help="Minibatch size")
  parser.add_argument("--iterations", "-t", type=int, default=100, help="Number of training iterations")
  parser.add_argument("--period", "-p", type=int, default=10, help="How often to look at objective")
  parser.add_argument("--parammin",  default=-2, type=float, help="minimum model parameter value")
  parser.add_argument("--parammax",  default=2, type=float, help="maximum model parameter value")
  parser.add_argument("--cliprate",  default=1, type=float, help="magnitude at which to clip")


  try:
    args = parser.parse_args()
  except IOError, msg:
    parser.error(str(msg))

  reader = codecs.getreader('utf8')
  writer = codecs.getwriter('utf8')
  infile = reader(args.infile)
  dictionaries = [reader(d) for d in args.dictionaries]
  

  # make initial transformation matrices
  # TODO: can resume! so turn this into a dict instead
  mats = []
  invmats = []
  paramrange=args.parammax-args.parammin
  matshape = (args.dim, args.dim)
  for i in xrange(args.langs):
    mat = np.matrix((np.random.rand(args.dim, args.dim)*paramrange)+args.parammin)
    mats.append(mat)
    invmats.append(LA.inv(mat))


  # TODO: would be cool if this could exist on-disk in some binary format so only the instructions need be passed in
  vocabs = [np.loadtxt(dictionary, dtype=str) for dictionary in dictionaries]
  vocab = dd(lambda: dict())
  for entry in np.vstack(vocabs):
    word = entry[0]
    lang = entry[1]
    vec = entry[2:].astype(float)
    if word not in vocab[lang]:
      vocab[lang][word]=vec

  print "loaded vocabularies"
  data = []
  for line in infile:
    inst = line.strip().split()
    inword = inst[0]
    inlang = inst[1]
    outlang = inst[2]
    outword = inst[3]
    # move language indices to the beginning of the vector
    data.append(np.concatenate((np.array([int(inlang), int(outlang)]), vocab[inlang][inword], vocab[outlang][outword]), axis=0))
  data = np.matrix(data)
  np.random.shuffle(data)
  devdata = data[:args.devsize]
  data = data[args.devsize:]
  if len(devdata) == 0:
    devdata = data
  print "loaded data"

  batchcount = data.shape[0]/args.minibatch # some data might be left but it's shuffled each time
  lastl2n2 = None
  for iteration in xrange(args.iterations):
    np.random.shuffle(data)
    for batchnum in xrange(batchcount):
#      print "batch "+str(batchnum)
      rowstart=batchnum*args.minibatch
      rowend = rowstart+args.minibatch
      langs = data[rowstart:rowend, :2].astype(int)
      all_batch_in = data[rowstart:rowend, 2:args.dim+2]
      all_batch_out = data[rowstart:rowend, args.dim+2:]
      # these are for training instances with the input language
      updates = [np.zeros(matshape) for x in xrange(args.langs)]
      # inverse; these are for training instances with the output language
      invupdates = [np.zeros(matshape) for x in xrange(args.langs)]
      # need to pick different matrices to update for each item so this will be slow
      for i in xrange(len(all_batch_in)):
        inlang=langs[i,0]
        outlang=langs[i,1]
        batch_in = all_batch_in[i]
        batch_out = all_batch_out[i]
        smat = mats[inlang]
        tmat = invmats[outlang]
        im = batch_in*smat
        imn= im*tmat
        delta = imn-batch_out
        print "Started with "+str(batch_in)
        print "Used "+str(mats[inlang])
        print "And "+str(invmats[outlang])
        print "Got "+str(imn)
        print "Wanted "+str(batch_out)
        print "Delta "+str(delta)
        # learning
        twodel = 2*delta
        invupdates[outlang] += (im.transpose()*twodel).transpose()
        updates[inlang] += batch_in.transpose()*twodel*tmat
        print "Inlang update "+str(updates[inlang])
        print "Outlang inv update "+str(invupdates[outlang])
      # minibatch over; do the updates
      # each matrix is first updated by its direct learning
      # it is also updated by the delta of the inverse of the update of its inverse
      # the inverse is then recalculated
      for l in xrange(args.langs):
        #        print "language "+str(l)
        # indirect update is the effect of updating the inverse and inverting vs. the original
        indirectupdate = LA.inv(invmats[l]-invupdates[l])-mats[l]
        print "indirect update for %d: %s" % (l, str(indirectupdate))
        print "direct update for %d: %s" % (l, str(updates[l]))
        totalupdate = indirectupdate-updates[l]

#        print mats[l].max(), mats[l].min(), invmats[l].max(), invmats[l].min()
        tunorm = LA.norm(totalupdate, ord=2)
        if tunorm > args.cliprate: # be able to set this
          totalupdate = args.cliprate*totalupdate/tunorm
          print "Clipped total update "+str(totalupdate)
        mats[l] = mats[l]+(totalupdate*args.learningrate/args.minibatch)

    if (iteration % args.period == 0): # report full training objective
      langs = devdata[:, :2].astype(int)
      all_batch_in = devdata[:, 2:args.dim+2]
      all_batch_out = devdata[:, args.dim+2:]
      l2n2 = 0
      allimn = []
      for i in xrange(len(all_batch_in)):
        inlang=langs[i,0]
        outlang=langs[i,1]
        batch_in = all_batch_in[i]
        batch_out = all_batch_out[i]
        smat = mats[inlang]
        tmat = invmats[outlang]
        imn = batch_in*smat*tmat
        allimn.append(imn)
        delta = imn-batch_out
        delnorm = LA.norm(delta, ord=2)
        l2n2 += delnorm*delnorm
        # TODO: find 0/1 nn accuracy and rank average
      allimn = np.vstack(allimn)
      sys.stdout.write("\nIteration "+str(iteration)+" = "+str(l2n2))# + 
      if lastl2n2 is not None:
        delta = l2n2-lastl2n2
        sys.stdout.write(" (delta: %f)" % delta)
      print " =\n" + str(allimn[:2,:10]) + "\nvs\n" + str(all_batch_out[:2,:10])
      lastl2n2=l2n2
  matsasdict = dict([(str(x[0]), x[1]) for x in enumerate(mats)])
  np.savez_compressed(args.modelfile, **matsasdict)

if __name__ == '__main__':
  main()

