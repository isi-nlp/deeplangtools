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

# suggestion from greg ver steeg: generate in I-space and then project to surface forms
# also, gaussian
# data form is word, language, vector
# to make artificial data easier to work with, assign same word label to input and output

def main():
  parser = argparse.ArgumentParser(description="Generate artificial data for the n matrix interlingus embedding experiment",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--edim", "-e", default=128, type=int, help="dimension of embeddings")
  parser.add_argument("--idim", "-i", default=128, type=int, help="dimension of interlingus")
  parser.add_argument("--langs", "-l", default=3, type=int, help="number of languages to generate data for")
  parser.add_argument("--size", "-s", default=1000, type=int, help="number of words to generate")
  parser.add_argument("--modelfile", "-m", help="all models file. Output file unless --additional is specified; then it's input")
  parser.add_argument("--additional", "-a", action='store_true', default=False, help="use already-generated models to make more data")
  parser.add_argument("--outprefix", "-o", default="random", help="prefix for output data files. Suffix is .edim.idim.lang")
  parser.add_argument("--parammin",  default=-2, type=float, help="minimum model parameter value")
  parser.add_argument("--parammax",  default=2, type=float, help="maximum model parameter value")
  parser.add_argument("--vecmin",  default=-1, type=float, help="minimum vector cell value")
  parser.add_argument("--vecmax",  default=1, type=float, help="maximum vector cell value")
  parser.add_argument("--std", default=0, type=float, help="std of gaussian noise. 0.1 is a good value if you want noise")


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
  # random inputs in interlingus space
  data = np.matrix((np.random.rand(args.size,args.idim)*vecrange)+args.vecmin)
  wordlabels = np.matrix(["w%d" % x for x in xrange(args.size)]).transpose()
  for i in xrange(args.langs):
    # generate surface form and write
    surface = data*(mats[str(i)].transpose())+np.random.normal(scale=args.std, shape=(args.size, args.edim))
    langid=np.matrix([str(i)]*args.size).transpose()
    outdata = np.concatenate((wordlabels, langid, surface.astype('|S20')), axis=1)
    outfile="%s.%d.%d.%d" % (args.outprefix, args.edim, args.idim, i)
    np.savetxt(outfile, outdata, fmt="%s") #, comments="# randomly generated %d x %d data; see %s and %s" % (args.size, args.edim, args.modelfile, args.lang2file))

    # double check parity with loaded model, in-mem vecs
    # TODO: turn this off when gaussian is added
    print "verifying"
    loadmats = np.load(args.modelfile)
    from scipy.spatial.distance import cosine
    recalc = outdata*loadmats[langid[0,0]]
    for index in xrange(args.size):
      cos = cosine(data[index], recalc[index])
      if cos > 0.0001:
        print "At item %d, cosine of %f for projection of %d" % (index, cos, i)
        sys.exit(1)
if __name__ == '__main__':
  main()

