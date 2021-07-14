#!/bin/bash

## GPU visibility
 export ROCR_VISIBLE_DEVICE=0
 export GPU_DEVICE_ORDINAL=0

## Boost
#export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

## Compiling
 export OLC_DEBUG_HIP_VERBOSE=1
 export OLC_DEBUG_HIP_DUMP=1
 export OLC_DEBUG_SAVE_TEMP_DIR=1

#make -j conv_driver_v2
#make -j conv_bwd_data_driver_v2
 make -j conv_driver_v2_olc

 rm -rf /root/_hip_binary_kernels_/
 rm -rf /tmp/olCompile*

LAYOUT=$1
ALGO=$2
VERIFY=$3
INIT=$4
LOG=$5
REPEAT=$6

./conv_driver_v2_olc      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	512	1024	1	1	14	14	1	1	1	1	0	0	0	0

################################# layout  algo  verify  init  log  repeat  N__ K___ C___ Y X Hi_ Wi__ Strides Dilations LeftPads RightPads
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	2048	1024	1	1	14	14	2	2	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	256	1024	1	1	14	14	1	1	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	512	1024	1	1	14	14	1	1	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	128	128	3	3	28	28	1	1	1	1	1	1	1	1
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	512	128	1	1	28	28	1	1	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	128	128	3	3	58	58	2	2	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	512	2048	1	1	7	7	1	1	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	1024	256	1	1	14	14	1	1	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	256	256	3	3	14	14	1	1	1	1	1	1	1	1
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	256	256	3	3	30	30	2	2	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	128	256	1	1	56	56	1	1	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	512	256	1	1	56	56	2	2	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	64	256	1	1	56	56	1	1	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	512	512	3	3	16	16	2	2	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	1024	512	1	1	28	28	2	2	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	128	512	1	1	28	28	1	1	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	256	512	1	1	28	28	1	1	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	2048	512	1	1	7	7	1	1	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	512	512	3	3	7	7	1	1	1	1	1	1	1	1
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	256	64	1	1	56	56	1	1	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	64	64	1	1	56	56	1	1	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	64	64	3	3	56	56	1	1	1	1	1	1	1	1

#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	64	3	7	7	230	230	2	2	1	1	0	0	0	0

#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	128	1024	1	1	17	17	1	1	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	192	1024	1	1	17	17	1	1	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	256	1024	1	1	17	17	1	1	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	384	1024	1	1	17	17	1	1	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	256	1536	1	1	8	8	1	1	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	384	1536	1	1	8	8	1	1	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	64	160	1	1	73	73	1	1	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	192	192	1	7	17	17	1	1	1	1	0	3	0	3
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	192	192	3	3	17	17	2	2	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	224	192	1	7	17	17	1	1	1	1	0	3	0	3
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	224	192	7	1	17	17	1	1	1	1	3	0	3	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	224	192	3	3	35	35	1	1	1	1	1	1	1	1
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	192	192	3	3	71	71	2	2	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	224	224	1	7	17	17	1	1	1	1	0	3	0	3
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	256	224	7	1	17	17	1	1	1	1	3	0	3	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	256	224	3	3	35	35	2	2	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	256	256	1	7	17	17	1	1	1	1	0	3	0	3
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	320	256	7	1	17	17	1	1	1	1	3	0	3	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	64	32	3	3	147	147	1	1	1	1	1	1	1	1
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	32	32	3	3	149	149	1	1	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	320	320	3	3	17	17	2	2	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	192	384	1	1	35	35	1	1	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	384	384	3	3	35	35	2	2	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	64	384	1	1	35	35	1	1	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	96	384	1	1	35	35	1	1	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	256	384	1	3	8	8	1	1	1	1	0	1	0	1
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	256	384	3	1	8	8	1	1	1	1	1	0	1	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	448	384	1	3	8	8	1	1	1	1	0	1	0	1
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	512	448	3	1	8	8	1	1	1	1	1	0	1	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	256	512	1	3	8	8	1	1	1	1	0	1	0	1
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	256	512	3	1	8	8	1	1	1	1	1	0	1	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	96	64	3	3	147	147	2	2	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	96	64	3	3	35	35	1	1	1	1	1	1	1	1
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	64	64	1	7	73	73	1	1	1	1	0	3	0	3
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	64	64	7	1	73	73	1	1	1	1	3	0	3	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	96	64	3	3	73	73	1	1	1	1	0	0	0	0
#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	96	96	3	3	35	35	1	1	1	1	1	1	1	1

#./conv_driver_v2      $LAYOUT $ALGO $VERIFY $INIT $LOG $REPEAT	256	32	3	3	3	299	299	2	2	1	1	0	0	0	0
