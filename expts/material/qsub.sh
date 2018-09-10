#!/bin/sh

# python get_write.py > write.sh
python $1 > write.sh

split -l 2 write.sh $2

for file in $2*; do qsub -e e -l mem_free=10G,ram_free=10G -M cnfxwang@gmail.com $file; done