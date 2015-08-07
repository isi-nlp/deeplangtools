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
  parser = argparse.ArgumentParser(description="Given MITRE titles file and vocabularies use some heuristics to filter down the list and output in instruction style",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--infile", "-i", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input titles file")
  parser.add_argument("--sourcedict", "-S", type=argparse.FileType('r'), help="source dictionary file")
  parser.add_argument("--targetdict", "-T", type=argparse.FileType('r'), help="target dictionary file")
  parser.add_argument("--sourcelang", "-s", default="spa", help="source language")
  parser.add_argument("--targetlang", "-t", default="eng", help="target language")
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

  for line in infile:
    src, trg = line.lstrip().rstrip().split(" ||| ")
    src = src.split()
    trg = trg.split()
    # singletons only
    if len(src) != 1 or len(trg) != 1:
      continue
    src = src[0]
    trg = trg[0]
    # no identities
    if src == trg:
      continue
    # no oov
    if src not in sourcedict or trg not in targetdict:
      continue
    # OTHER HEURISTICS...
    outfile.write("%s %s %s %s\n" % (src, args.sourcelang, args.targetlang, trg))
if __name__ == '__main__':
  main()

