#!/bin/sh
g++ -std=c++11 deblur.cpp -o deblur `pkg-config --cflags opencv` `pkg-config --libs opencv`

g++ -std=c++11 mask2color.cpp -o mask2color `pkg-config --cflags opencv` `pkg-config --libs opencv`

