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
from scipy.spatial.distance import cosine
from scipy.spatial import cKDTree as kdt
from sklearn.preprocessing import normalize
import cPickle
import heapq

scriptdir = os.path.dirname(os.path.abspath(__file__))

def main():
  parser = argparse.ArgumentParser(description="Evaluate the 2 matrix embedding experiment",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--sourcedictionary", "-S", type=argparse.FileType('r'),  help="source vocabulary dictionary of the form lang word vec; headed by row col; need only be relevant to the evaluation set")
  parser.add_argument("--targetdictionary", "-T", type=argparse.FileType('r'),  help="target vocabulary dictionary (pickled)")
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

  sourceinfo = map(int, sourcedictionary.readline().strip().split())
  sourcedim = sourceinfo[1]

  smat = np.matrix(np.load(args.modelfile)['s'])
  tmat = np.matrix(np.load(args.modelfile)['t'])
  print smat.shape
  print tmat.shape
  
  # load transformation matrices
  # TODO: would be cool if this could exist on-disk in some binary format so only the instructions need be passed in
  # Kludgy: store source and target in different structures
  vocab = dd(lambda: dict())
  # for kdt lookup
  pretargets, targetvoc = cPickle.load(args.targetdictionary)
  targets = kdt(pretargets)
  invtargetvoc = dict()
  for key, word in enumerate(targetvoc):
    invtargetvoc[word]=key
  print len(targetvoc)
  
  try:
    for ln, line in enumerate(sourcedictionary):
      entry = line.strip().split(' ')
      if len(entry) < sourcedim+2:
        sys.stderr.write("skipping line %d in %s because it only has %d fields; first field is %s\n" % (ln, sourcedictionary.name, len(entry), entry[0]))
        continue
      lang = entry[0]
      word = ' '.join(entry[1:-sourcedim])
      vec = np.array(entry[-sourcedim:]).astype(float)
      vocab[lang][word]=vec
  except:
    print sourcedictionary.name
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
    xform = np.asarray(invec*smat*tmat)[0]
    neighbors = []
    cosines, cands = targets.query(xform, args.nbest)
    for cos, cand in zip(cosines, cands):
      neighbors.append((cos, targetvoc[cand]))
    report = inst[:4]
    nb_words = [x[1] for x in neighbors]
    #cosines: xform to truth, xform to 1best, truth to 1best
    truth=pretargets[invtargetvoc[outword]]
    onebest=pretargets[invtargetvoc[nb_words[0]]]
    xtruth=str(cosine(xform, truth))
    if len(nb_words) > 0:
      xbest=str(cosine(xform, onebest))
      truthbest=str(cosine(truth, onebest))
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

