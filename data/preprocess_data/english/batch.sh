#!/bin/bash

for id in `seq 0 7`
do
    echo "begin to search for worker: $id"
    ./phrase_split.sh $id &
done
