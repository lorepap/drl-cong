#!/bin/bash

base_path=$(pwd) #be sure you're in the src dir
# # Build kernel header file
# echo '--- Building kernel header file ---'
# echo ''
# cd $base_path/kernel && bash init.sh || exit 1


cd $base_path/kernel
echo ''

# Build and insert kernel module
echo '--- Building and inserting kernel module file ---'
echo ''
sudo chmod +xrw $base_path/kernel/build.sh
sh build.sh || exit 1


echo ''
# Set mimic as congestion control protocol
echo '-- Set mimic as congestion control protocol'
echo ''
sudo sysctl net.ipv4.tcp_congestion_control=mimic || exit 1