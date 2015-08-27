#! /usr/bin/env python

# try to make the inner loop faster by assembling data before mat mult
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


def main():
  parser = argparse.ArgumentParser(description="Do gradient descent for the n matrix interlingus embedding experiment",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--dictionaries", "-d", nargs='+', type=argparse.FileType('r'), help="vocabulary dictionaries of the form word lang vec")
  parser.add_argument("--idim", "-I", default=128, type=int, help="dimension of interlingus")
  parser.add_argument("--infile", "-i", type=argparse.FileType('r'), default=sys.stdin, help="training instruction of the form word1 lang1 lang2 word2")
  parser.add_argument("--devsize", "-s", type=int, default=0, help="How much dev data to sample; rest is training")
  parser.add_argument("--modelfile", "-f", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="all models output file")
  parser.add_argument("--learningrate", "-r", type=float, default=0.001, help="Learning rate")
  parser.add_argument("--minibatch", "-m", type=int, default=10, help="Minibatch size")
  parser.add_argument("--iterations", "-t", type=int, default=100, help="Number of training iterations")
  parser.add_argument("--period", "-p", type=int, default=10, help="How often to look at objective")
  parser.add_argument("--parammin",  default=-2, type=float, help="minimum model parameter value")
  parser.add_argument("--parammax",  default=2, type=float, help="maximum model parameter value")
  parser.add_argument("--cliprate", "-c", default=1, type=float, help="magnitude at which to clip")
  parser.add_argument("--noearly", action='store_true', default=False, help="no early stopping")
  parser.add_argument("--seed", default=None, type=int, help="set the seed for deterministic behavior")
  parser.add_argument("--dumpdir", default=None, help="directory to dump incremental models; no dump if not specified")

  try:
    args = parser.parse_args()
  except IOError, msg:
    parser.error(str(msg))

  np.random.seed(args.seed) # TODO: is this the right way to go?

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
  dictionaries = [reader(d) for d in args.dictionaries]
  dicts_by_lang = dd(list)
  langdims = dict()

  for d in dictionaries:
    info = d.readline().strip().split()
    dims = int(info[1])
    lang = info[2]
    if lang in langdims:
      if dims != langdims[lang]:
        raise ValueError("Multiple dimensions seen for %s: %d and %d" % (lang, dims, langdims[lang]))
    else:
      langdims[lang]=dims
    dicts_by_lang[lang].append(d)
  interdim = args.idim

  # make initial transformation matrices
  paramrange=args.parammax-args.parammin
  inmats = {}
  outmats = {}
  # TODO: would be cool if this could exist on-disk in some binary format so only the instructions need be passed in
  vocab = dd(lambda: dict())
  for l in langdims.keys():
    inmats[l] = np.matrix((np.random.rand(langdims[l], interdim)*paramrange)+args.parammin)
    outmats[l] = np.matrix((np.random.rand(langdims[l], interdim)*paramrange)+args.parammin).transpose()
    fdim = langdims[l]
    for dfile in dicts_by_lang[l]:
      try:
        for ln, line in enumerate(dfile):
          entry = line.strip().split(' ')
          if len(entry) < fdim+1:
            sys.stderr.write("skipping line %d in %s because it only has %d fields; first field is %s\n" % (ln, dfile.name, len(entry), entry[0]))
            continue
          word = ' '.join(entry[:-fdim])
          vec = np.array(map(float, entry[-fdim:]))
          if word not in vocab[l]:
            vocab[l][word]=vec
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
    data.append((inlang, outlang, 
                 vocab[inlang][inword], 
                 vocab[outlang][outword]))
  print "Skipped %d for missing vocab; data has %d entries" % (skipped, len(data))
  vocab.clear()
  print "Cleared vocab"
  print len(data)
  np.random.shuffle(data)
  devdata = data[:args.devsize]
  data = data[args.devsize:]
  if len(devdata) == 0:
    devdata = data
  print "loaded data"

  batchcount = len(data)/args.minibatch # some data might be left but it's shuffled each time
  lastl2n2=None
  for iteration in xrange(args.iterations):
    np.random.shuffle(data)
    for batchnum in xrange(batchcount):
      rowstart=batchnum*args.minibatch
      rowend = rowstart+args.minibatch
      all_batch = data[rowstart:rowend]
      in_updates = {}
      out_updates = {}
      in_counts = dd(int)
      out_counts = dd(int)
      for l, dim in langdims.iteritems():
        in_updates[l] = np.zeros((dim, interdim))
        out_updates[l] = np.zeros((interdim, dim))
      # need to pick different matrices to update for each item so this will be slow
      perlangs = dd(lambda: dd(list))
      for inlang, outlang, invec, outvec in all_batch:
        in_counts[inlang]+=1
        out_counts[outlang]+=1
        # could this be done in a better way so the matrices don't have to be teased apart later?
        perlangs[inlang][outlang].append((invec, outvec))
      for inlang in perlangs.keys():
        for outlang in perlangs[inlang].keys():          
          smat = inmats[inlang]
          tmat = outmats[outlang]
          inmat = np.matrix([x[0] for x in perlangs[inlang][outlang]])
          outmat = np.matrix([x[1] for x in perlangs[inlang][outlang]])
          im = inmat*smat
          imn = im*tmat
          twodel = 2*(imn-outmat)
          out_updates[outlang] += im.transpose()*twodel
          in_updates[inlang] += inmat.transpose()*twodel*tmat.transpose()
      for l in langdims.keys():
        for counts, updates, mats in zip([in_counts, out_counts], [in_updates, out_updates], [inmats, outmats]):
          if counts[l] > 0:
            update = updates[l]/counts[l]
            if args.cliprate > 0:
              norm = LA.norm(update, ord=2)
              if norm > args.cliprate:
                update = args.cliprate*update/norm
            mats[l] = mats[l]-(update*args.learningrate)
    if (iteration % args.period == 0): # report full training objective
      if args.dumpdir is not None:
        matsasdict = {}
        for l in langdims.keys():
          matsasdict['%s_in' % l]=inmats[l]
          matsasdict['%s_out' % l]=outmats[l]
          np.savez_compressed(os.path.join(args.dumpdir, "%d.model" % iteration), **matsasdict)
      l2n2 = 0
      for inlang, outlang, invec, outvec in devdata:
        smat = inmats[inlang]
        tmat = outmats[outlang]
        im = invec*smat
        imn = im*tmat
        delta = imn-outvec
        delnorm = LA.norm(delta, ord=2)
        l2n2 += delnorm*delnorm
      print "iteration "+str(iteration)+": "+str(l2n2)
      if args.noearly:
        pass
      elif lastl2n2 is not None and l2n2 >= lastl2n2:
        print "Stopping early"
        break
      lastl2n2=l2n2
  matsasdict = {}
  for l in langdims.keys():
    matsasdict['%s_in' % l]=inmats[l]
    matsasdict['%s_out' % l]=outmats[l]
  np.savez_compressed(args.modelfile, **matsasdict)

if __name__ == '__main__':
  main()

