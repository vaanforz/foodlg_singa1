# Remove all containers
#docker stop $(docker ps -a -q)
#docker rm $(docker ps -a -q)

# Remove all images
#docker rmi $(docker images -q)

cd ..
docker build -t rs/app .
cd nginx_files
docker build -t reverseproxy .
docker-compose up --scale test=4