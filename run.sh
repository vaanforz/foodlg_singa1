#!/bin/bash
set -e

echo -n "Please enter link for Rafiki's predictor host: "
read predictor_host
endpoint='/predict'
link="$predictor_host$endpoint"
echo $link > rafiki_predictor_host.txt

chmod 755 docker_entrypoint.sh
docker run -p 8085:5000 -v $PWD:/root/foodlg -d foodlg_image