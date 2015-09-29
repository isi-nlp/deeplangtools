#! /usr/bin/env python
import argparse
import sys
import codecs
from collections import defaultdict as dd
import re
import os.path
scriptdir = os.path.dirname(os.path.abspath(__file__))


def main():
  parser = argparse.ArgumentParser(description="Form three-way tuples a b c from a-b, a-c, b-c",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--centerfile", "-c", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="center mapping file; should be a->b, probably not used in actual training previously")
  parser.add_argument("--leftfile", "-l", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="left mapping file; should be a->c")
  parser.add_argument("--rightfile", "-r", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="right mapping file; should be b->c")
  parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="output file")



  try:
    args = parser.parse_args()
  except IOError, msg:
    parser.error(str(msg))

  reader = codecs.getreader('utf8')
  writer = codecs.getwriter('utf8')
  leftfile = reader(args.leftfile)
  rightfile = reader(args.rightfile)
  centerfile = reader(args.centerfile)
  outfile = writer(args.outfile)

  lefts = dd(set)
  rights = dd(set)
  
  # used to make sure languages are consistent
  leftlang = set()
  rightlang = set()
  centerlang = set()

  for line in leftfile:
    lw, ll, cl, cw = line.strip().split()
    leftlang.add(ll)
    centerlang.add(cl)
    lefts[lw].add(cw)
  for line in rightfile:
    rw, rl, cl, cw = line.strip().split()
    rightlang.add(rl)
    centerlang.add(cl)
    rights[rw].add(cw)
  for line in centerfile:
    lw, ll, rl, rw = line.strip().split()
    leftlang.add(ll)
    rightlang.add(rl)
    for cw in list(lefts[lw].intersection(rights[rw])):
      cl = list(centerlang)[0]
      outfile.write("%s %s %s %s %s %s\n" % (lw, ll, rw, rl, cw, cl))
  if len(leftlang) != 1 or len(rightlang) != 1 or len(centerlang) != 1:
    sys.stderr.write("WARNING: wrong number of languages for left: "+str(leftlang)+", right: "+str(rightlang)+", center: "+str(centerlang)+", results probably wrong\n")

if __name__ == '__main__':
  main()
