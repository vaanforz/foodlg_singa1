#!/bin/bash
set -e
redis-server db.conf  &
python app.py  &
python server.py --model xception --dataset food204 --imgwidth 299 --imgheight 299 --weights /root/model-zoo/xception-24-0.84.h5 --topk 5

