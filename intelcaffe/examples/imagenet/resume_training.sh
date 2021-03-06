#!/usr/bin/env sh
set -e

./build/tools/caffe train \
    --solver=models/bvlc_reference_caffenet/solver.prototxt \
    --snapshot=models/bvlc_reference_caffenet/caffenet_train_iter_3529.solverstate \
    $@
