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
import pickle
scriptdir = os.path.dirname(os.path.abspath(__file__))

def main():
  parser = argparse.ArgumentParser(description="pickle up dictionary for fast querying later",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--infile", "-i", type=argparse.FileType('r'), default=sys.stdin, help="vector dictionary")
  parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('wb'), default=sys.stdout, help="pickled binary")
  try:
    args = parser.parse_args()
  except IOError as msg:
    parser.error(str(msg))


  infile = args.infile
  info = infile.readline().strip().split()
  dim = int(info[1])
  lang = info[2]
  # word->vec hash (for source)
  vocab = dict()
  # parallel wordlist, veclist (for target)
  targets = []
  targetvoc = []
  try:
    for ln, line in enumerate(infile):
      entry = line.strip().split(' ')
      if len(entry) < dim+1:
        sys.stderr.write("skipping line %d in %s because it only has %d fields; first field is %s\n" % (ln, infile.name, len(entry), entry[0]))
        continue
      word = ' '.join(entry[:-dim])
      vec = np.array(entry[-dim:]).astype(float)
      targets.append(vec)
      targetvoc.append(word)
      vocab[word] = vec
  except:
    print(infile.name)
    print(line)
    print(len(entry))
    print(word)
    print(ln)
    raise
  # normalize for euclidean distance nearest neighbor => cosine with constant
  # targets = kdt(normalize(np.array(targets), axis=1, norm='l2'))
  data = {'lang':lang, 'dim':dim, 'vocab':vocab, 'targets':targets, 'targetvoc':targetvoc}
  pickle.dump(data, args.outfile, pickle.HIGHEST_PROTOCOL)
if __name__ == '__main__':
  main()

