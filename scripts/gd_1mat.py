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
import os
import errno

scriptdir = os.path.dirname(os.path.abspath(__file__))

# TODO: auto-read dimensions in other scripts
# TODO: reverse other dictionaries to be lang word vec

def main():
  parser = argparse.ArgumentParser(description="Do gradient descent for the 1 matrix interlingus embedding experiment",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--sourcedictionary", "-S", type=argparse.FileType('r'),  help="source vocabulary dictionary of the form lang word vec; headed by row col")
  parser.add_argument("--targetdictionary", "-T", type=argparse.FileType('r'),  help="target vocabulary dictionary of the form lang word vec; headed by row col")
  parser.add_argument("--infile", "-i", type=argparse.FileType('r'), default=sys.stdin, help="training instruction of the form word1 lang1 lang2 word2")
  parser.add_argument("--devsize", "-s", type=int, default=0, help="How much dev data to sample; rest is training")
  parser.add_argument("--modelfile", "-f", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="all models output file")
  parser.add_argument("--learningrate", "-r", type=float, default=0.001, help="Learning rate")
  parser.add_argument("--minibatch", "-m", type=int, default=10, help="Minibatch size")
  parser.add_argument("--iterations", "-t", type=int, default=100, help="Number of training iterations")
  parser.add_argument("--dumpdir", "-d", default=None, help="directory to dump incremental models; no dump if not specified")
  parser.add_argument("--period", "-p", type=int, default=10, help="How often to look at objective/dump model")
  parser.add_argument("--parammin",  default=-2, type=float, help="minimum model parameter value")
  parser.add_argument("--parammax",  default=2, type=float, help="maximum model parameter value")
  parser.add_argument("--cliprate", "-c", default=0, type=float, help="magnitude at which to clip")
  parser.add_argument("--noearly", action='store_true', default=False, help="no early stopping")

  try:
    args = parser.parse_args()
  except IOError, msg:
    parser.error(str(msg))

  if args.dumpdir is not None:
    # http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    try:
        os.makedirs(args.dumpdir)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(args.dumpdir):
            pass
        else: raise

  reader = codecs.getreader('utf8')
  writer = codecs.getwriter('utf8')
  infile = reader(args.infile)
  sourcedictionary = reader(args.sourcedictionary)
  targetdictionary = reader(args.targetdictionary)
  
  sourceinfo = sourcedictionary.readline().strip().split()
  targetinfo = targetdictionary.readline().strip().split()
  sourcelang=sourceinfo[2]
  targetlang=targetinfo[2]
  dims = {}
  dims[sourcelang] = int(sourceinfo[1])
  dims[targetlang] = int(targetinfo[1])
  dicts_by_lang = {}
  dicts_by_lang[sourcelang]=sourcedictionary
  dicts_by_lang[targetlang]=targetdictionary
  sourcedim = dims[sourcelang]
  targetdim = dims[targetlang]
  print sourcedim,targetdim
  # make initial transformation matrix

  paramrange=args.parammax-args.parammin

  mat = np.matrix((np.random.rand(sourcedim, targetdim)*paramrange)+args.parammin)

  # TODO: would be cool if this could exist on-disk in some binary format so only the instructions need be passed in
  vocab = dd(lambda: dict())
  for lang in (sourcelang, targetlang):
    fdim = dims[lang]
    dfile = dicts_by_lang[lang]
    try:
      for ln, line in enumerate(dfile):
        entry = line.strip().split(' ')
        if len(entry) < fdim+1:
          sys.stderr.write("skipping line %d in %s because it only has %d fields; first field is %s\n" % (ln, dfile.name, len(entry), entry[0]))
          continue
        word = ' '.join(entry[:-fdim])
        vec = np.array(entry[-fdim:])
        if word not in vocab[lang]:
          vocab[lang][word]=vec
    except:
      print dfile.name
      print line
      print len(entry)
      print word
      print ln
      raise

  print "loaded vocabularies"
  data = []
  skipped = 0
  for line in infile:
    inst = line.strip().split()
    inword = inst[0]
    inlang = inst[1]
    outlang = inst[2]
    outword = inst[3]
    if inword not in vocab[inlang]:
      skipped+=1
      continue
    if outword not in vocab[outlang]:
      skipped+=1
      continue
    # move language indices to the beginning of the vector
    data.append(np.concatenate((np.array([inlang, outlang]), vocab[inlang][inword], vocab[outlang][outword]), axis=0))
  print "Skipped %d for missing vocab; data has %d entries" % (skipped, len(data))
  vocab.clear()
  print "Cleared vocab"
  print len(data)
  data = np.matrix(data)
  np.random.shuffle(data)
  devdata = data[:args.devsize]
  data = data[args.devsize:]
  if len(devdata) == 0:
    devdata = data
  print "loaded data"

  batchcount = data.shape[0]/args.minibatch # some data might be left but it's shuffled each time
  lastl2n2=None
  for iteration in xrange(args.iterations):
    np.random.shuffle(data)
    for batchnum in xrange(batchcount):
      rowstart=batchnum*args.minibatch
      rowend = rowstart+args.minibatch
      batch_in = data[rowstart:rowend, 2:sourcedim+2].astype(float)
      batch_out = data[rowstart:rowend, sourcedim+2:].astype(float)
      im = batch_in*mat
      twodel = 2*(im-batch_out)
      update = batch_in.transpose()*twodel/args.minibatch
      if args.cliprate > 0:
        norm = LA.norm(update, ord=2)
        if norm > args.cliprate:
          update = args.cliprate*update/norm
      mat = mat-(update*args.learningrate)
    if (iteration % args.period == 0): # report full training objective
      if args.dumpdir is not None:
        np.savez_compressed(os.path.join(args.dumpdir, "%d.model" % iteration), mat)
      batch_in = devdata[:, 2:sourcedim+2].astype(float)
      batch_out = devdata[:, sourcedim+2:].astype(float)
      im = batch_in*mat
      delta = im-batch_out
      delnorm = LA.norm(delta, ord=2)
      l2n2 = delnorm*delnorm
      print "iteration "+str(iteration)+": "+str(l2n2) + " = " + str(im[:2,:10]) + " vs " + str(batch_out[:2,:10])
      if args.noearly:
        pass
      elif lastl2n2 is not None and l2n2 >= lastl2n2:
        print "Stopping early"
        break
      lastl2n2=l2n2
  np.savez_compressed(args.modelfile, mat)

if __name__ == '__main__':
  main()

