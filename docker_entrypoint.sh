#!/bin/bash
set -e
redis-server db.conf  &
python app.py