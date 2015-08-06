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
  parser = argparse.ArgumentParser(description="Random generation of instruction file",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--langs", "-l", default=3, type=int, help="number of languages to learn for")
  parser.add_argument("--vocabulary", "-v", default=1000, type=int, help="number of words in vocabulary")
  parser.add_argument("--size", "-s", type=int, default=10, help="Amount of data to generate")
  parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="instruction file")


  try:
    args = parser.parse_args()
  except IOError, msg:
    parser.error(str(msg))

  reader = codecs.getreader('utf8')
  writer = codecs.getwriter('utf8')
  outfile = writer(args.outfile)
  for i in xrange(args.size):
    v = np.random.randint(0, high=args.vocabulary)
    il, ol = np.random.permutation(range(args.langs))[:2]
    outfile.write("w%d %d %d w%d\n" % (v, il, ol, v))

if __name__ == '__main__':
  main()

