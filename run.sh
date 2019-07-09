#!/bin/bash
set -e

docker run -p 8085:5000 -v /hdd1/yisen/server-front-sep/database:/root/database -v /hdd1/yisen/server-front-sep/foodlg_singa1/:/root/foodlg -d foodlg-gpu


