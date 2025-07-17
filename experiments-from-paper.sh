#!/bin/sh

conda init bash
conda activate mocahesp2

for solver in bo casmo bounce; do
  for seed in {1..10}; do
    for objective in ackley20c antibody maxsat28 labs cco maxsat125; do
      python test-mocahesp.py -f $objective --solver $solver -n 800 --seed $seed
    done
    for objective in ackley20c labs; do
      python test-mocahesp.py -f $objective --solver $solver -n 800 --seed $seed --shifted
    done
  done
done