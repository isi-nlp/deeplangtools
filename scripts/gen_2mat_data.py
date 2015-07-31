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

def main():
  parser = argparse.ArgumentParser(description="Generate artificial data for the two matrix interlingus embedding experiment",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--edim", "-e", default=128, type=int, help="dimension of embeddings")
  parser.add_argument("--idim", "-i", default=128, type=int, help="dimension of interlingus")
  parser.add_argument("--size", "-s", default=1100, type=int, help="number of (s, t) pairs to generate")
  parser.add_argument("--modelafile", "-a", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="model a output file")
  parser.add_argument("--modelbfile", "-b", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="model b output file")
  parser.add_argument("--lang1file", "-1", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="L1 output file")
  parser.add_argument("--lang2file", "-2", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="L2 output file")
  parser.add_argument("--parammin",  default=-2, type=float, help="minimum model parameter value")
  parser.add_argument("--parammax",  default=2, type=float, help="maximum model parameter value")
  parser.add_argument("--vecmin",  default=-1, type=float, help="minimum vector cell value")
  parser.add_argument("--vecmax",  default=1, type=float, help="maximum vector cell value")



  try:
    args = parser.parse_args()
  except IOError, msg:
    parser.error(str(msg))

  # make gold transformation matrices
  paramrange=args.parammax-args.parammin
  smat = np.matrix((np.random.rand(args.edim, args.idim)*paramrange)+args.parammin)
  tmat = np.matrix((np.random.rand(args.edim, args.idim)*paramrange)+args.parammin).transpose()

  vecrange=args.vecmax-args.vecmin
  strain_in = np.matrix((np.random.rand(args.size,args.edim)*vecrange)+args.vecmin)
  strain_out = strain_in*smat*tmat
  
  np.savetxt(args.lang1file, strain_in) #, comments="# randomly generated %d x %d data; see %s and %s" % (args.size, args.edim, args.modelfile, args.lang2file))
  np.savetxt(args.lang2file, strain_out) #, comments="# randomly generated %d x %d data; see %s and %s" % (args.size, args.edim, args.modelfile, args.lang2file))
  np.savetxt(args.modelafile, smat) #, comments="# randomly generated %d x %d data; see %s and %s" % (args.size, args.edim, args.modelfile, args.lang2file))
  np.savetxt(args.modelbfile, tmat) #, comments="# randomly generated %d x %d data; see %s and %s" % (args.size, args.edim, args.modelfile, args.lang2file))
if __name__ == '__main__':
  main()

