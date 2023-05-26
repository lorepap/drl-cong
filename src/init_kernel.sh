#!/bin/bash

base_path=/home/lorenzo/Desktop/drl-cong/src
# # Build kernel header file
# echo '--- Building kernel header file ---'
# echo ''
# cd $base_path/kernel && bash init.sh || exit 1


cd $base_path/kernel
echo ''

# Build and insert kernel module
echo '--- Building and inserting kernel module file ---'
echo ''
sh build.sh || exit 1


echo ''
# Set mimic as congestion control protocol
echo '-- Set mimic as congestion control protocol'
echo ''
sudo sysctl net.ipv4.tcp_congestion_control=mimic || exit 1