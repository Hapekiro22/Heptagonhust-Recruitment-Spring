#!/bin/bash

# Load Intel Vtune
echo "Load Intel Vtune"
eval $(spack load --sh intel-oneapi-vtune)
vtune --version

