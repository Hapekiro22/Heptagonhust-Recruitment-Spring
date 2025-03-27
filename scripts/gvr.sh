#!/bin/bash

# ./getVt_results.sh <version> <input_file>

version=$1

if [ "$2" == "1" ]; then
    #Use vgg16.conf
    CONFIG_FILE="vgg16"
elif [ "$2" == "0" ]; then
    #Use small.conf
    CONFIG_FILE="small"
else
    echo "Wrong Parameters!"
fi

directory="vtune_results/$version-$CONFIG_FILE"

eval $(spack load --sh intel-oneapi-vtune)

vtune-backend --data-directory $directory