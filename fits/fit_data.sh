#!/bin/bash

MAX_JOBS=32

submit() {
  while [ "$(jobs -p | wc -l)" -ge "$MAX_JOBS" ]; do
    sleep 1
  done

  OMP_NUM_THREADS=1 python fit_data.py $1 $2 $3 $4 --n_jobs=4 &
}


for obs in ggl rsd
do
  for cosmo in 0 1 4 13 100 101 102 103 104 105 112 113 116 117 118 119 120 125 126 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146
  do
    submit $obs $cosmo --assembly_bias_mock --assembly_bias_model
    submit $obs $cosmo --assembly_bias_model
    submit $obs $cosmo --assembly_bias_mock
  done
done
wait
