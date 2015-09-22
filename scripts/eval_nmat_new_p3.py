#! /usr/bin/env python
import argparse
import sys
import codecs

from collections import defaultdict as dd
import re
import os.path
import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import cosine
from scipy.spatial import cKDTree as kdt
from sklearn.preprocessing import normalize
import pickle

scriptdir = os.path.dirname(os.path.abspath(__file__))

def main():
  parser = argparse.ArgumentParser(description="Evaluate the n matrix embedding experiment",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--dictionaries", "-d", nargs='+', type=argparse.FileType('r'), default=[sys.stdin,], help="vocabulary dictionaries of the form word vec with a header")
  parser.add_argument("--infile", "-i", type=argparse.FileType('r'), default=sys.stdin, help="evaluation instruction of the form word1 lang1 lang2 [word2]. If word2 is absent it is only predicted, not evaluated")
  parser.add_argument("--modelfiles", "-m", nargs='+', default=[], help="all models input files")
  parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="results file of the form word1 lang1 lang2 word2 [pos wordlist], where the first three fields are identical to eval and the last field is the 1-best prediction. If truth is known, ordinal position of correct answer (-1 if not found) followed by the n-best list in order")
  parser.add_argument("--nbest", "-n", type=int, default=10, help="nbest neighbors generated for purposes of evaluation")
  parser.add_argument("--pickle", "-p", action='store_true', default=False, help="dictionaries are pickled with pickle_vocab")
  parser.add_argument("--hidewords", action='store_true', default=False, help="don't actually print nbest words")


  try:
    args = parser.parse_args()
  except IOError as msg:
    parser.error(str(msg))


  infile = args.infile
  dictionaries = [pickle.load(d) for d in args.dictionaries] if args.pickle else [d for d in args.dictionaries]
  dicts_by_lang = dd(list)
  langdims = dict()
  outfile = args.outfile
  for d in dictionaries:
    if args.pickle:
      lang = d['lang']
      dims = int(d['dim'])
    else:
      info = d.readline().strip().split()
      dims = int(info[1])
      lang = info[2]
    if lang in langdims:
      if dims != langdims[lang]:
        raise ValueError("Multiple dimensions seen for %s: %d and %d" % (lang, dims, langdims[lang]))
    else:
      langdims[lang]=dims
    dicts_by_lang[lang].append(d)
  inmats = {}
  outmats = {}
  vocab = dd(lambda: dict())
  # for kdt lookup
  targets = dd(list)
  targetvoc = dd(list)
  models = [ np.load(x) for x in args.modelfiles ]
  for l in list(langdims.keys()):
    inmats[l] = [ np.matrix(x['%s_in' % l]) for x in models ]
    outmats[l] = [ np.matrix(x['%s_out' % l]) for x in models ]
    fdim = langdims[l]
    for dfile in dicts_by_lang[l]:
      if args.pickle:
        print("Unpickling for "+l)
        vocab[l].update(dfile['vocab'])
        targets[l].extend(dfile['targets'])
        targetvoc[l].extend(dfile['targetvoc'])
      else:
        print("processing "+dfile.name)
        try:
          for ln, line in enumerate(dfile):
            entry = line.strip().split(' ')
            if len(entry) < fdim+1:
              sys.stderr.write("skipping line %d in %s because it only has %d fields; first field is %s\n" % (ln, dfile.name, len(entry), entry[0]))
              continue
            word = ' '.join(entry[:-fdim])
            vec = np.array(entry[-fdim:]).astype(float)
  #          print "Adding "+l+" -> "+word
            vocab[l][word]=vec
            targets[l].append(vec)
            targetvoc[l].append(word)
        except:
          print(dfile.name)
          print(line)
          print(len(entry))
          print(word)
          print(ln)
          raise
    # normalize for euclidean distance nearest neighbor => cosine with constant
    targets[l] = kdt(normalize(np.array(targets[l]), axis=1, norm='l2'))
  print("loaded vocabularies")

  for line in infile:
    inst = line.strip().split()
    inword = inst[0]
    inlang = inst[1]
    outlang = inst[2]
    outword = inst[3] if len(inst) > 3 else None
    if inword not in vocab[inlang]:
      sys.stderr.write("Warning: Couldn't find %s -> %s\n" % (inlang, inword))
      continue
    report = inst[:4]
    invec = np.matrix(vocab[inlang][inword])
    for smat, tmat in zip(inmats[inlang], outmats[outlang]):
      xform = np.asarray(invec*smat*tmat)[0]
      neighbors = []
      cosines, cands = targets[outlang].query(xform, args.nbest)
      for cos, cand in zip(cosines, cands):
        neighbors.append((cos, targetvoc[outlang][cand]))
      nb_words = [x[1] for x in neighbors]
  #    print nb_words
      #cosines: xform to truth, xform to 1best, truth to 1best
      if outword in vocab[outlang]:
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
        if not args.hidewords:
          report.extend(nb_words)
    outfile.write('\t'.join(report)+"\n")
  # TODO: some overall stats to stderr?

if __name__ == '__main__':
  main()

