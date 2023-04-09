trg_emotions=$@

echo "Run video with emotions: $trg_emotions"

bash ./preprocess.sh actor
python manipulator/test.py --celeb actor --checkpoints_dir ./manipulator_checkpoints --trg_emotions $trg_emotions --exp_name result
bash ./postprocess.sh actor result checkpoints_actor
python postprocessing/images2video.py --imgs_path actor/result/full_frames --out_path new_actor.mp4 --audio actor/videos/actor.mp4