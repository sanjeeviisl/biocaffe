#!/usr/bin/env sh
set -e

LMDB=/hdd/plant_lmdb_data/lmdb
DATA=/hdd/plant_lmdb_data/data
TOOLS=build/tools

TRAIN_DATA_ROOT=/hdd/plant_lmdb_data/normalization/
VAL_DATA_ROOT=/hdd/plant_lmdb_data/validation/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

# Set ENCODE=true to encode the images as compressed JPEGs stored in the LMDB.
# Leave as false for uncompressed (raw) images.
ENCODE=false
if $ENCODE; then
  ENCODE_FLAG='--encoded=true'
  ENCODE_TYPE_FLAG='--encode_type=jpg'
else
  ENCODE_FLAG='--encoded=false'
  ENCODE_TYPE_FLAG=''
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $ENCODE_FLAG \
    $ENCODE_TYPE_FLAG \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $LMDB/imagenet_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $ENCODE_FLAG \
    $ENCODE_TYPE_FLAG \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/val.txt \
    $LMDB/imagenet_val_lmdb

echo "Done."
