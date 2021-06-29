#!/bin/bash

## GPU visibility
 export ROCR_VISIBLE_DEVICE=0
 export GPU_DEVICE_ORDINAL=0

#make -j conv_driver
 make -j conv_driver_v2

LAYOUT=$1
ALGO=$2
VERIFY=$3
INIT=$4
LOG=$5
REPEAT=$6

######################  layout  algo  verify  init  log  repeat  N__ K___ C___ Y X Hi_ Wi__ Strides Dilations LeftPads RightPads
#driver/conv_driver_v2 $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  128  256  192 3 3  71   71     2 2       1 1      1 1       1 1
#driver/conv_driver_v2 $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  128  256 1024 1 7  17   17     1 1       1 1      0 3       0 3
 #driver/conv_driver_v2 $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  256 2048 3 3  14   14     1 1       1 1      1 1       1 1
driver/conv_driver_v2 $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  256  256 3 3  14   14     1 1       1 1      1 1       1 1
#driver/conv_driver_v2 $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  512  512 3 3   7    7     1 1       1 1      1 1       1 1

#driver/conv_driver_v2 $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  128  512  192 3 3  35   35     2 2       1 1      0 0       0 0
#driver/conv_driver_v2 $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  256  256 3 3  30   30     2 2       1 1      0 0       0 0
#driver/conv_driver_v2 $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  512  512 3 3  16   16     2 2       1 1      0 0       0 0

#driver/conv_driver_v2 $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256 2048 1024 1 1  14   14     2 2       1 1      0 0       0 0
#driver/conv_driver_v2 $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  256 1024 1 1  14   14     1 1       1 1      0 0       0 0
#driver/conv_driver_v2 $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT  256  512 2048 1 1   7    7     1 1       1 1      0 0       0 0
