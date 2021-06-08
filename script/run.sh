#!/bin/bash

#make -j conv_driver
 make -j conv_driver_v2

LAYOUT=$1
VERIFY=$2
INIT=$3
LOG=$4
REPEAT=$5

######################  layout  verify  init  log  repeat  N__ K__ C__ Y X Hi_ Wi__ Strides Dilations LeftPads RightPads
 driver/conv_driver_v2 $LAYOUT $VERIFY $INIT $LOG $REPEAT  128 128 192 3 3  71   71     2 2       1 1      1 1       1 1
