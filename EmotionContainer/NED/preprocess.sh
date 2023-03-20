celeb=$1

python preprocessing/detect.py --celeb $celeb --save_videos_info --save_full_frames
python preprocessing/eye_landmarks.py --celeb $celeb --align
python preprocessing/segment_face.py --celeb $celeb
python preprocessing/reconstruct.py --celeb $celeb --save_shapes --save_nmfcs
python preprocessing/align.py --celeb $celeb --faces_and_masks --shapes --nmfcs --landmarks


