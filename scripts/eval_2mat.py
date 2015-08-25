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
from scipy.spatial.distance import cosine
from scipy.spatial import cKDTree as kdt
from sklearn.preprocessing import normalize
from lshash import LSHash

import heapq

scriptdir = os.path.dirname(os.path.abspath(__file__))

def main():
  parser = argparse.ArgumentParser(description="Evaluate the 2 matrix embedding experiment",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--sourcedictionary", "-S", type=argparse.FileType('r'),  help="source vocabulary dictionary of the form lang word vec; headed by row col")
  parser.add_argument("--targetdictionary", "-T", type=argparse.FileType('r'),  help="target vocabulary dictionary of the form lang word vec; headed by row col")
  parser.add_argument("--bits", "-b", default=10, type=int, help="number of bits for hash projection. higher = more accurate knn (and slower)")

  parser.add_argument("--infile", "-i", type=argparse.FileType('r'), default=sys.stdin, help="evaluation instruction of the form word1 lang1 lang2 [word2]. If word2 is absent it is only predicted, not evaluated")
  parser.add_argument("--modelfile", "-m", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="all models input file")
  parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="results file of the form word1 lang1 lang2 word2 [pos wordlist], where the first three fields are identical to eval and the last field is the 1-best prediction. If truth is known, ordinal position of correct answer (-1 if not found) followed by the n-best list in order")
  parser.add_argument("--nbest", "-n", type=int, default=10, help="nbest neighbors generated for purposes of evaluation")
  parser.add_argument("--doSlow", action='store_true', default=False, help="slow n^2 calculation of nearest neighbors")


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


  smat = np.matrix(np.load(args.modelfile)['s'])
  tmat = np.matrix(np.load(args.modelfile)['t'])
  print smat.shape
  print tmat.shape
  # make nn indices for each language
  # Create a random binary hash
  rbp = RandomBinaryProjections('rbp', args.bits)
  # create engine for target language
  engine = Engine(targetdim, lshashes=[rbp], distance=CosineDistance(), vector_filters=[NearestFilter(args.nbest)])
  
  # load transformation matrices
  # TODO: would be cool if this could exist on-disk in some binary format so only the instructions need be passed in
  # Kludgy: store source and target in different structures
  vocab = dd(lambda: dict())
  targets = []
  targetvoc = []
  lshes = LSHash(20, targetdim)
  for istarget, fdim, dfile in zip((False, True), (sourcedim, targetdim), (sourcedictionary, targetdictionary)):
    try:
      for ln, line in enumerate(dfile):
        entry = line.strip().split(' ')
        if len(entry) < fdim+2:
          sys.stderr.write("skipping line %d in %s because it only has %d fields; first field is %s\n" % (ln, dfile.name, len(entry), entry[0]))
          continue
        lang = entry[0]
        word = ' '.join(entry[1:-fdim])
        vec = np.array(entry[-fdim:]).astype(float)
        vocab[lang][word]=vec
        if istarget:
          engine.store_vector(vec, word)
          targets.append(vec)
          targetvoc.append(word)
          lshes.index(vec, extra_data=word)
    except:
      print dfile.name
      print line
      print len(entry)
      print word
      print ln
      raise
  targets = kdt(normalize(np.array(targets), axis=1, norm='l2'))
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
    xform = np.asarray(invec*smat*tmat)[0]
    # TODO: n^2 or the other exact way to get nbest; engine seems broken
    if args.doSlow:
      neighbors = []

#      lshresults = lshes.query(xform, num_results=args.nbest, distance_func="cosine")
#      print lshresults
#      for res in lshresults:
#        neighbors.append((res[1], res[0][1]))
      cosines, cands = targets.query(xform, args.nbest)

      for cos, cand in zip(cosines, cands):
        neighbors.append((cos, targetvoc[cand]))

      # cmps = [] # make heap
      # heapq.heapify(cmps)
      # for word, tvec in vocab[outlang].iteritems():
      #   heapq.heappush(cmps, (cosine(xform, tvec), word))
      # neighbors = []
      # for i in xrange(args.nbest):
      #   neighbors.append(heapq.heappop(cmps))
    else:
      neighbors = engine.neighbours(xform)
    report = inst[:4]
    nb_words = [x[1] for x in neighbors]
    #cosines: xform to truth, xform to 1best, truth to 1best
    truth=vocab[outlang][outword]
    xtruth=str(cosine(xform, truth))
    if len(nb_words) > 0:
      xbest=str(cosine(xform, vocab[outlang][nb_words[0]]))
      truthbest=str(cosine(truth, vocab[outlang][nb_words[0]]))
    else:
      xbest="???"
      truthbest="???"
    rank = nb_words.index(outword) if outword in nb_words else -1
    report.append(str(rank))
    report.extend([xtruth, xbest, truthbest])
    report.extend(nb_words)
    outfile.write('\t'.join(report)+"\n")
  # TODO: some overall stats to stderr?

if __name__ == '__main__':
  main()

