#! /bin/bash

IMAGES="../images"
BUILD="../build"

for IMAGE in $IMAGES/cat*.png; do
  for time in `seq 0 25`; do
    echo $IMAGE
    $BUILD/global_mem $IMAGE
    $BUILD/shared_mem $IMAGE
    $BUILD/constant_mem $IMAGE
  done
done

