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
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.filters import NearestFilter
from nearpy.distances import CosineDistance

scriptdir = os.path.dirname(os.path.abspath(__file__))

def main():
  parser = argparse.ArgumentParser(description="Evaluate the n matrix inverse model interlingus embedding experiment",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--dim", "-e", default=128, type=int, help="dimension of embeddings/interlingus")
  parser.add_argument("--langs", "-l", default=3, type=int, help="number of languages to generate data for")
  parser.add_argument("--bits", "-b", default=10, type=int, help="number of bits for hash projection. higher = more accurate knn (and slower)")
  parser.add_argument("--dictionaries", "-d", nargs='+', type=argparse.FileType('r'), default=[sys.stdin,], help="vocabulary dictionaries of the form word lang vec")
  parser.add_argument("--infile", "-i", type=argparse.FileType('r'), default=sys.stdin, help="evaluation instruction of the form word1 lang1 lang2 [word2]. If word2 is absent it is only predicted, not evaluated")
  parser.add_argument("--modelfile", "-m", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="all models input file")
  parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="results file of the form word1 lang1 lang2 word2 [pos wordlist], where the first three fields are identical to eval and the last field is the 1-best prediction. If truth is known, ordinal position of correct answer (-1 if not found) followed by the n-best list in order")
  parser.add_argument("--nbest", "-n", type=int, default=10, help="nbest neighbors generated for purposes of evaluation")


  try:
    args = parser.parse_args()
  except IOError, msg:
    parser.error(str(msg))


  reader = codecs.getreader('utf8')
  writer = codecs.getwriter('utf8')
  infile = reader(args.infile)
  dictionaries = [reader(d) for d in args.dictionaries]
  outfile = writer(args.outfile)
  
  # make nn indices for each language
  # Create a random binary hash
  rbp = RandomBinaryProjections('rbp', args.bits)
  # create engines for each language
  engines = [Engine(args.dim, lshashes=[rbp], distance=CosineDistance(), vector_filters=[NearestFilter(args.nbest)]) for x in xrange(args.langs)]
  
  # load transformation matrices
  mats = np.load(args.modelfile)
  invmats = {}
  for name, mat in mats.items():
    invmats[name]=LA.inv(mat)
  vocabs = [np.loadtxt(dictionary, dtype=str) for dictionary in dictionaries]
  vocab = dd(lambda: dict())
  for entry in np.vstack(vocabs):
    word = entry[0]
    lang = entry[1]
    vec = entry[2:].astype(float)
    if word not in vocab[lang]:
      vocab[lang][word]=vec
      engines[int(lang)].store_vector(vec, word)

  for line in infile:
    inst = line.strip().split()
    inword = inst[0]
    inlang = inst[1]
    outlang = inst[2]
    outword = inst[3] if len(inst) > 3 else None
    if inword not in vocab[inlang]:
      sys.stderr.write("Warning: Couldn't find %s -> %s\n" % (inlang, inword))
      continue
    invec = np.matrix(vocab[inlang][inword])
    xform = np.asarray(invec*mats[inlang]*invmats[outlang])[0]
    neighbors = engines[int(outlang)].neighbours(xform)
    report = inst[:3]
    report.append(neighbors[0][1])
    if outword is not None:
      # # show cosine of truth with cosine of nbest
      # from scipy.spatial.distance import cosine
      # outvec = vocab[outlang][outword]
      # costruth = cosine(xform, outvec)
      # if costruth > 0.00001:
      #   print "Cosine with truth of %f" % costruth
      #   print xform[:10]
      #   print outvec[:10]
#      print "Cosine with truth is %f; cosine with guess is %f\n" % (cosine(xform, outvec), neighbors[0][2])
      nb_words = [x[1] for x in neighbors]
      rank = nb_words.index(outword) if outword in nb_words else -1
      report.append(str(rank))
      report.extend(nb_words)
    outfile.write('\t'.join(report)+"\n")
  # TODO: some overall stats to stderr?

if __name__ == '__main__':
  main()

