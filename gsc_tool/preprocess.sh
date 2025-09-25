SCENE=Bartender
BASE_DIR=data/GSC

python gsc_tool/video_preprocess.py \
    --scene $SCENE --base_dir $BASE_DIR/$SCENE

python gsc_tool/gen_poses_bds_file.py \
    --scene $SCENE --base_dir $BASE_DIR/$SCENE --frame_num 1

python gsc_tool/run_per_frame_colmap.py \
    --scene $SCENE --base_dir $BASE_DIR/$SCENE --frame_num 1

python gsc_tool/name_adapter.py \
    --scene $SCENE --base_dir $BASE_DIR/$SCENE