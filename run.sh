#!/bin/bash
set -e

chmod 755 docker_entrypoint.sh
docker run -p 8085:5000 -v $PWD:/root/foodlg -d foodlg_image_no_gpu