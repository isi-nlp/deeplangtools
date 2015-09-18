#! /usr/bin/env python
import argparse
import sys
import codecs
from itertools import izip
from collections import defaultdict as dd
import re
import os.path
scriptdir = os.path.dirname(os.path.abspath(__file__))


def main():
  parser = argparse.ArgumentParser(description="Given LRLP lexicon and vocabularies use some heuristics to filter down the list and output in instruction style",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--infile", "-i", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input lexicon file")
  parser.add_argument("--sourcedict", "-S", type=argparse.FileType('r'), help="source dictionary file")
  parser.add_argument("--targetdict", "-T", type=argparse.FileType('r'), help="target dictionary file")
  parser.add_argument("--sourcelang", "-s", default="spa", help="source language")
  parser.add_argument("--targetlang", "-t", default="eng", help="target language")
  parser.add_argument("--includeident", "-I", action='store_true', default=False, help="allow identity mappings to be included")
  parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="output instruction file")



  try:
    args = parser.parse_args()
  except IOError, msg:
    parser.error(str(msg))

  reader = codecs.getreader('utf8')
  writer = codecs.getwriter('utf8')
  infile = reader(args.infile)
  sdfile = reader(args.sourcedict)
  tdfile = reader(args.targetdict)
  outfile = writer(args.outfile)

  sourcedict = set()
  targetdict = set()
  for line in sdfile:
    sourcedict.add(line.strip())
  for line in tdfile:
    targetdict.add(line.strip())

  identities = 0
  oov = 0
  mword = 0
  for line in infile:
    try:
      src, pos, trgs = line.lstrip().rstrip().split("\t")
    except:
      sys.stderr.write("Bad line: "+line)
      continue
    src = re.sub(r'\([^\(\)]+\)', '', src).split()
    # src singletons only
    if len(src) != 1:
      mword+=1
      continue
    src = src[0].lower()
    # no src oov
    if src not in sourcedict:
      oov+=1
      continue
    # get rid of parentheticals and split on commas or semicolons
    trgs = re.split(r'[;,]', re.sub(r'\([^\(\)]+\)', '', trgs.lower()))
    for trg in trgs:
      trg = trg.strip()
      # no multiwords or empties
      if len(trg.split()) != 1:
        mword+=1
        continue
      # no identities
      if (not args.includeident) and src == trg:
        identities+=1
        continue
      # no trg oov
      if trg not in targetdict:
        oov+=1
        continue
      # OTHER HEURISTICS...
      outfile.write("%s %s %s %s\n" % (src, args.sourcelang, args.targetlang, trg))
  sys.stderr.write("%d ident %d oov %d mword\n" % (identities, oov, mword))
if __name__ == '__main__':
  main()

