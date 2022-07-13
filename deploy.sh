#!/bin/bash
rm nn_build/nn_build
rm -rf nn_build/build
rm -rf nn_build/src
mkdir nn_build/src
cp -r src.out/* nn_build/src/
cd nn_build
make
