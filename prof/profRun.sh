#!/bin/bash

echo "$1 = ["
for n in 5000 10000 15000 20000 25000 50000 75000 100000 250000 500000 1000000; do
  build/cis565_boids $n 2> /dev/null
done
echo "]"
