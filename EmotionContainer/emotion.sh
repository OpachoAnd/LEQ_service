celeb=$1
emotions=$(cat emotions.txt)



echo "Build Docker"
sudo docker build . -t ned -f NED/Dockerfile

echo "Run Docker"
sudo docker run --name ned_container -t ned $emotions

echo "Ready?"