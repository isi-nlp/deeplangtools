#! /usr/bin/env python
import argparse
import sys
import codecs
from itertools import izip
from collections import defaultdict as dd
import re
import os.path
import numpy as np
scriptdir = os.path.dirname(os.path.abspath(__file__))

# data form is word, language, vector
# to make artificial data easier to work with, assign same word label to input and output

def main():
  parser = argparse.ArgumentParser(description="Generate artificial data for the n matrix interlingus embedding experiment",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--edim", "-e", default=128, type=int, help="dimension of embeddings")
  parser.add_argument("--idim", "-i", default=128, type=int, help="dimension of interlingus")
  parser.add_argument("--langs", "-l", default=3, type=int, help="number of languages to generate data for")
  parser.add_argument("--size", "-s", default=1100, type=int, help="number of (s, t) pairs to generate")
  parser.add_argument("--modelfile", "-m", help="all models file. Output file unless --additional is specified; then it's input")
  parser.add_argument("--additional", "-a", action='store_true', default=False, help="use already-generated models to make more data")
  parser.add_argument("--lang1file", "-1", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="side 1 output file")
  parser.add_argument("--lang2file", "-2", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="side 2 output file")
  parser.add_argument("--parammin",  default=-2, type=float, help="minimum model parameter value")
  parser.add_argument("--parammax",  default=2, type=float, help="maximum model parameter value")
  parser.add_argument("--vecmin",  default=-1, type=float, help="minimum vector cell value")
  parser.add_argument("--vecmax",  default=1, type=float, help="maximum vector cell value")



  try:
    args = parser.parse_args()
  except IOError, msg:
    parser.error(str(msg))

  # make gold transformation matrices

  if args.additional:
    mats = np.load(args.modelfile)
  else:
    mats = {}
    paramrange=args.parammax-args.parammin
    for i in xrange(args.langs):
      mats[str(i)] = np.matrix((np.random.rand(args.edim, args.idim)*paramrange)+args.parammin)
    np.savez_compressed(args.modelfile, **mats)

  
  vecrange=args.vecmax-args.vecmin
  # random inputs
  strain_in = np.matrix((np.random.rand(args.size,args.edim)*vecrange)+args.vecmin)
  # assign them to languages and assign target languages
  langmap = np.matrix(np.random.randint(0, high=args.langs, size=(len(strain_in),2)))
  wordlabels = np.matrix(["w%d" % x for x in xrange(len(strain_in))]).transpose()
  strain_out = []
  try:
    for i in xrange(len(strain_in)):
      # hack: if we generate the same languages as input and output re-sample until this is not true
      while langmap[i,0] == langmap[i,1]:
        langmap[i] = np.random.randint(0, high=args.langs, size=2)
      elem = strain_in[i]*mats[str(langmap[i,0])]*(mats[str(langmap[i,1])].transpose())
      strain_out.append(np.asarray(elem))
    strain_out = np.vstack(strain_out)
#    print strain_in
#    print strain_out
    strain_in = np.concatenate((wordlabels, langmap[:,0], strain_in.astype('|S20')), axis=1)
    strain_out = np.concatenate((wordlabels, langmap[:,1], np.matrix(strain_out).astype('|S20')), axis=1)
#    print strain_in
#    print strain_out
  except:
    print i
    raise
  np.savetxt(args.lang1file, strain_in, fmt="%s") #, comments="# randomly generated %d x %d data; see %s and %s" % (args.size, args.edim, args.modelfile, args.lang2file))
  np.savetxt(args.lang2file, strain_out, fmt="%s") #, comments="# randomly generated %d x %d data; see %s and %s" % (args.size, args.edim, args.modelfile, args.lang2file))




  # double check parity with loaded model, in-mem vecs
  print "verifying"
  loadmats = np.load(args.modelfile)
  from scipy.spatial.distance import cosine
  for (index, (ei, eo)) in enumerate(zip(strain_in, strain_out)):
    recalc = ei[:,2:].astype(float)*loadmats[ei[0,1]]*(loadmats[eo[0,1]].transpose())
    cos = cosine(eo[:,2:].astype(float), recalc)
    if cos > 0.0001:
      print "At item %d, cosine of %f between %s and %s" % (index, cos, eo[:,2:6], recalc[:,:4])
      sys.exit(1)
if __name__ == '__main__':
  main()

