#!/usr/bin/env bash
make clean build

# make run ARGS="-input=lena.pgm"
make run ARGS="-input=data/Lena-grey.pgm -angle=12 -angleStep=13 -output=data/Lena_12_13.pgm"