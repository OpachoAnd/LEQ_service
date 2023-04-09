celeb=$1
trg_emotions=$(cat /home/malinux/NED/emotions.txt)


echo "Copy weights..."
rm -rf ./NED/checkpoints_actor
mkdir ./NED/checkpoints_actor
if [ "$celeb" = "pacino" ]
then
    cp -a /home/malinux/NED/actors/checkpoints_pacino/. ./NED/checkpoints_actor/
elif [ "$celeb" = "roberts" ]
then
    cp -a /home/malinux/NED/actors/checkpoints_roberts/. ./NED/checkpoints_actor/
elif [ "$celeb" = "tarantino" ]
then
    cp -a /home/malinux/NED/actors/checkpoints_tarantino/. ./NED/checkpoints_actor/
else
    echo "Invalid mode given"
fi
cp -a /home/malinux/NED/weights/DECA/data/. ./NED/DECA/data/
cp -a /home/malinux/NED/weights/manipulator_checkpoints/. ./NED/manipulator_checkpoints/
cp -a /home/malinux/NED/weights/preprocessing/segmentation/. ./NED/preprocessing/segmentation/

echo "Copy video..."
rm -rf ./NED/actor
mkdir ./NED/actor
cp -a /home/malinux/NED/video/. ./NED/actor/videos/

echo "Build Docker"
docker build . -t ned -f NED/Dockerfile 

echo "Run Docker"
docker run --name ned_container --gpus all -t ned $trg_emotions
docker cp ned_container:/app/new_actor.mp4  /home/malinux/NED
echo "Ready!"
