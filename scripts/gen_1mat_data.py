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

# NOTE: prepends output with row col

def main():
  parser = argparse.ArgumentParser(description="Generate artificial data for the one matrix no-interlingus embedding experiment",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--sdim", "-S", default=128, type=int, help="dimension of source embeddings")
  parser.add_argument("--tdim", "-T", default=128, type=int, help="dimension of target embeddings")
  parser.add_argument("--length", "-l", default=1100, type=int, help="number of (s, t) pairs to generate")
  parser.add_argument("--modelfile", "-m", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="model output file")
  parser.add_argument("--lang1file", "-1", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="L1 output file")
  parser.add_argument("--lang1", "-s", help="Language 1")
  parser.add_argument("--lang2", "-t", help="Language 2")
  parser.add_argument("--lang2file", "-2", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="L2 output file")
  parser.add_argument("--parammin",  default=-2, type=float, help="minimum model parameter value")
  parser.add_argument("--parammax",  default=2, type=float, help="maximum model parameter value")
  parser.add_argument("--vecmin",  default=-1, type=float, help="minimum vector cell value")
  parser.add_argument("--vecmax",  default=1, type=float, help="maximum vector cell value")
  parser.add_argument("--std", "-n", default=0, type=float, help="std of gaussian noise. 0.01 is a good value if you want noise")


  try:
    args = parser.parse_args()
  except IOError, msg:
    parser.error(str(msg))

  # make gold transformation matrices
  paramrange=args.parammax-args.parammin
  mat = np.matrix((np.random.rand(args.sdim, args.tdim)*paramrange)+args.parammin)
  np.savez_compressed(args.modelfile, mat)

  vecrange=args.vecmax-args.vecmin
  wordlabels = np.matrix(["w%d" % x for x in xrange(args.length)]).transpose()
  slanglabels = np.matrix([args.lang1]*args.length).transpose()
  tlanglabels = np.matrix([args.lang2]*args.length).transpose()
  strain_in = np.matrix((np.random.rand(args.length,args.sdim)*vecrange)+args.vecmin)
  strain_out = strain_in*mat
  if args.std > 0:
    strain_out+=np.random.normal(scale=args.std, size=(args.length, args.tdim))

  ssize= np.matrix([map(str, [args.length, args.sdim])+['']*args.sdim])
  tsize= np.matrix([map(str, [args.length, args.tdim])+['']*args.tdim])
  strain_in = np.concatenate((ssize, np.concatenate((slanglabels, wordlabels, strain_in.astype('|S20')), axis=1)), axis=0)
  strain_out = np.concatenate((tsize, np.concatenate((tlanglabels, wordlabels,  strain_out.astype('|S20')), axis=1)), axis=0)


  np.savetxt(args.lang1file, strain_in, fmt="%s") #, comments="# randomly generated %d x %d data; see %s and %s" % (args.size, args.edim, args.modelfile, args.lang2file))
  np.savetxt(args.lang2file, strain_out, fmt="%s") #, comments="# randomly generated %d x %d data; see %s and %s" % (args.size, args.edim, args.modelfile, args.lang2file))

if __name__ == '__main__':
  main()

