#!/bin/sh
#parse v0.4 lagecy autograd profiler result (intel-pytorch)
#v0.5 don't need so much trouble
echo "profiling $1"
sed -i "s/us/ /g" "$1"
cat $1 | awk '{a[$1]+=$2; b[$1]+=$3; c[$1]+=$4; d[$1]+=$5; e[$1]+=$6} END {for(name in a) print name, a[name]/c[name],b[name],c[name],d[name]/1000,e[name]}' | sed 's/[ ][ ]*/,/g'| tee profile.csv
