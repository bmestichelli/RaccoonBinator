#!/bin/bash
ulimit -s unlimited
ulimit -l unlimited
ulimit -v unlimited

# Use GPU 0 (or 1 if you prefer)
export CUDA_VISIBLE_DEVICES=0

export OMP_NUM_THREADS=8
export OMP_STACKSIZE=16GB

arrayA=( 50000.  )

for mcl in ${arrayA[*]};
do
echo "$mcl"
mkdir -p s"$mcl"
cd s"$mcl"

cp /mnt/sdc/bmestichelli/tests_pop3_massive/mcluster ./
cp /mnt/sdc/bmestichelli/test_raccoonBinator/RaccoonBinator-main/*.py ./

#taskset -c 0-7 ./mcluster -M "$mcl" -R 1 -P 1 -W 6 -C 5 -G 1 -f 2 -a -0.74 -a -0.17 -m 0.08 -m 20 -m 300 -b 0.4 -e 0 -t 1 -u 1 #top-heavy IMF
#taskset -c 0-7 ./mcluster -M "$mcl" -R 1 -P 1 -W 6 -C 5 -G 1 -f 2 -a -1.3 -a -2.3 -m 0.08 -m 0.5 -m 300 -b 0.4 -e 0 -t 1 -u 1 #Kroupa IMF
taskset -c 0-7 ./mcluster -M "$mcl" -R 1 -P 1 -W 6 -C 5 -G 1 -f 2 -a -1 -m 0.08 -m 300 -b 0.4 -e 0 -t 1 -u 1  #log-flat IMF

python recomBinaries.py > logbin

cd /mnt/sdc/bmestichelli/test_raccoonBinator/RaccoonBinator-main

done
