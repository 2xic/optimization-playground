#!/bin/bash
start=$1
end="$(($start + 10))"
cuda=$2
echo "$2"
if [ -z "$2" ]
  then
    echo "No cuda device specified"
    exit 0
fi

for i in $(seq $start $end)
do
   CUDA_VISIBLE_DEVICES=$cuda python3 train.py $i
done
