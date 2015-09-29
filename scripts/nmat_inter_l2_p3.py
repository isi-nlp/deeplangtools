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
  parser = argparse.ArgumentParser(description="Show l2norm of all pairwise languages in a trained model",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--dictionaries", "-d", nargs='+', type=argparse.FileType('r'), default=[sys.stdin,], help="vocabulary dictionaries of the form word vec with a header")
  parser.add_argument("--infile", "-i", type=argparse.FileType('r'), default=sys.stdin, help="evaluation instruction of the form word1 lang1 word2 lang2 ... wordn langn.")
  parser.add_argument("--modelfiles", "-m", nargs='+', default=[], help="all models input files")
  parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="for each model, for each pairwise language, the l2norm")
  parser.add_argument("--pickle", "-p", action='store_true', default=False, help="dictionaries are pickled with pickle_vocab")

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


  data = dd(list)
  langmap = {}
  for line in infile:
    linedata = line.strip().split()
    for dset, (word, lang) in enumerate(zip(linedata[::2], linedata[1::2])):
      if word not in vocab[lang]:
        sys.stderr.write("Warning: Couldn't find %s -> %s\n" % (lang, word))
        continue
      if dset not in langmap:
        langmap[dset]=lang
      elif langmap[dset] != lang:
        sys.stderr.write("Language collision at %d: %s vs %s\n" % (dset, lang, landmap[dset]))
        sys.exit(1)
      data[dset].append(vocab[lang][word])
  for lang in langmap:
    data[lang] = np.matrix(data[lang])
  langs = len(langmap.keys())
  for m, mfile in enumerate(args.modelfiles):
    for d1 in range(langs):
      l1 = langmap[d1]
      d1xform = data[d1]*inmats[l1][m]
      for d2 in range(d1+1, langs):
        l2 = langmap[d2]
        if l2 in inmats:
          # i-i calculation
          d2xform = data[d2]*inmats[l2][m]
          delta = d1xform-d2xform
          delnorm = LA.norm(delta, ord=2)
          l2n2 = delnorm*delnorm
          outfile.write("%s\tii\t%s\t%s\t%f\n" % (mfile, l1, l2, l2n2))
        if l2 in outmats:
          xform = d1xform*outmats[l2][m]
          delta = xform-data[d2]
          delnorm = LA.norm(delta, ord=2)
          l2n2 = delnorm*delnorm
          outfile.write("%s\tio\t%s\t%s\t%f\n" % (mfile, l1, l2, l2n2))
if __name__ == '__main__':
  main()

