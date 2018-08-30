"""Find the unlabeled files"""

import os
import sys


def find_unlabeled_files(dir):
  num =  0
  for filename in os.listdir(dir):
    found = False
    with open(os.path.join(dir, filename)) as file:
      for line in file.readlines():
        if '<GOV>' in line or '<LIF>' in line:
          found = True
          continue
    if not found:
      print(os.path.join(dir, filename))
      num+=1
  print(num)

def main():
  dirs = sys.argv[1:]
  for dir in dirs:
    find_unlabeled_files(dir)


if __name__ == '__main__':
  main()
