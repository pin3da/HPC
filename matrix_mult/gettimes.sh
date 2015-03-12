#! /bin/bash

sort -s -V times > timesorted
awk '{print $1}' < timesorted > sizes
awk '{print $2}' < timesorted > serial
awk '{print $3}' < timesorted > parallel
