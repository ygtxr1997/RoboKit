#!/bin/bash


# 配置变量
FILE_NAMES=(
#  "0627_pot_source"
#  "0704_pepper_source"
#  "0709_coffee_source"
  "0709_coffee_distractor_rand"
  "0709_coffee_table_rand"
  "0709_coffee_object_rand"
  "0709_coffee_light_rand"
  "0704_pepper_distractor_rand"
  "0704_pepper_table_rand"
  "0704_pepper_object_rand"
  "0704_pepper_light_rand"
  "0627_pot_table_rand"
)

for FILE_NAME in "${FILE_NAMES[@]}"; do

  python scripts/01_preprocess_data.py  \
          -R "/home/geyuan/local_soft/TCL/${FILE_NAME}/"

done
