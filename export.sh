#!/usr/bin/env bash

./build.sh

docker save picai_baseline_nndetection_semi_supervised_processor | gzip -c > picai_baseline_nndetection_semi_supervised_processor_1.0.tar.gz
