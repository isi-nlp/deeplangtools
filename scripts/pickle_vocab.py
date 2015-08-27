#! /usr/bin/env python
import argparse
import sys
import codecs
import re
import os.path
import numpy as np
from numpy import linalg as LA
from scipy.spatial import cKDTree as kdt
from sklearn.preprocessing import normalize
import cPickle
scriptdir = os.path.dirname(os.path.abspath(__file__))

def main():
  parser = argparse.ArgumentParser(description="pickle up target dictionary for fast querying later",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--infile", "-i", type=argparse.FileType('r'), default=sys.stdin, help="evaluation instruction of the form word1 lang1 lang2 [word2]. If word2 is absent it is only predicted, not evaluated")
  parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('wb'), default=sys.stdout, help="results file of the form word1 lang1 lang2 word2 [pos wordlist], where the first three fields are identical to eval and the last field is the 1-best prediction. If truth is known, ordinal position of correct answer (-1 if not found) followed by the n-best list in order")
  try:
    args = parser.parse_args()
  except IOError, msg:
    parser.error(str(msg))


  reader = codecs.getreader('utf8')
  writer = codecs.getwriter('utf8')
  infile = reader(args.infile)
  info = map(int, infile.readline().strip().split())
  dim = info[1]

  # for kdt lookup
  targets = []
  targetvoc = []
  try:
    for ln, line in enumerate(infile):
      entry = line.strip().split(' ')
      if len(entry) < dim+2:
        sys.stderr.write("skipping line %d in %s because it only has %d fields; first field is %s\n" % (ln, infile.name, len(entry), entry[0]))
        continue
      word = ' '.join(entry[1:-dim])
      vec = np.array(entry[-dim:]).astype(float)
      targets.append(vec)
      targetvoc.append(word)
  except:
    print infile.name
    print line
    print len(entry)
    print word
    print ln
    raise
  # normalize for euclidean distance nearest neighbor => cosine with constant
  targets = kdt(normalize(np.array(targets), axis=1, norm='l2'))
  data = [targets, targetvoc]
  cPickle.dump(data, args.outfile, cPickle.HIGHEST_PROTOCOL)
if __name__ == '__main__':
  main()

