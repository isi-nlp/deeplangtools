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
  parser = argparse.ArgumentParser(description="Evaluate the 1 matrix no interlingus embedding experiment",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--sourcedictionary", "-S", type=argparse.FileType('r'),  help="source vocabulary dictionary of the form lang word vec; headed by row col")
  parser.add_argument("--targetdictionary", "-T", type=argparse.FileType('r'),  help="target vocabulary dictionary of the form lang word vec; headed by row col")
  parser.add_argument("--bits", "-b", default=10, type=int, help="number of bits for hash projection. higher = more accurate knn (and slower)")

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
  outfile = writer(args.outfile)
  sourcedictionary = reader(args.sourcedictionary)
  targetdictionary = reader(args.targetdictionary)
  
  sourceinfo = map(int, sourcedictionary.readline().strip().split())
  targetinfo = map(int, targetdictionary.readline().strip().split())
  sourcedim = sourceinfo[1]
  targetdim = targetinfo[1]
  print sourcedim,targetdim


  mat = np.matrix(np.load(args.modelfile)['arr_0'])
  # make nn indices for each language
  # Create a random binary hash
  rbp = RandomBinaryProjections('rbp', args.bits)
  # create engine for target language
  engine = Engine(targetdim, lshashes=[rbp], distance=CosineDistance(), vector_filters=[NearestFilter(args.nbest)])
  
  # load transformation matrices
  # TODO: would be cool if this could exist on-disk in some binary format so only the instructions need be passed in
  vocab = dd(lambda: dict())
  for fdim, dfile in zip((sourcedim, targetdim), (sourcedictionary, targetdictionary)):
    try:
      for ln, line in enumerate(dfile):
        entry = line.strip().split(' ')
        if len(entry) < fdim+2:
          sys.stderr.write("skipping line %d in %s because it only has %d fields; first field is %s\n" % (ln, dfile.name, len(entry), entry[0]))
          continue
        lang = entry[0]
        word = ' '.join(entry[1:-fdim])
        vec = np.array(entry[-fdim:]).astype(float)
        if word not in vocab[lang]:
          vocab[lang][word]=vec
          engine.store_vector(vec, word)
    except:
      print dfile.name
      print line
      print len(entry)
      print word
      print ln
      raise

  print "loaded vocabularies"

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
    xform = np.asarray(invec*mat)[0]
    neighbors = engine.neighbours(xform)
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

