#!/bin/bash
set -e

docker run -p 8085:5000 -v $PWD:/root/foodlg -d foodlg-gpu



