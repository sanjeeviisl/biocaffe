#!/usr/bin/env sh

LMDB=/hdd/plant_lmdb_data/lmdb
DATA=/hdd/plant_lmdb_data/data
TOOLS=build/tools
#/hdd/lmdb_data/lmdb/imagenet_val_lmdb

$TOOLS/compute_image_mean $LMDB/imagenet_val_lmdb \
  $DATA/imagenet_mean.binaryproto

echo "Done."
