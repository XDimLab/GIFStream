# Set the directory containing the scenes
SCENE_DIR="/data/dataset/mpeg"
# Set the directory to store results
RESULT_DIR="/data/gscodec/GIFStream_branch"
# Set the rendering trajectory path
RENDER_TRAJ_PATH="ellipse"
# List of scenes to process
SCENE_LIST="Bartender"
# List of entropy lambda values (rate-distortion tradeoff parameter)
ENTROPY_LAMBDA_LIST=(0.0005 0.001 0.002 0.004)
# Data factor for training
DATA_FACTOR=1
# Number of frames per GOP (Group of Pictures)
GOP=65
# The index of the first frame to process
FIRST_FRAME=50
# Total number of frames to process
TOTAL_FRAME=65
TYPE=GSC

# Loop over each scene in the scene list
for SCENE in $SCENE_LIST;
do

    # Loop over each entropy lambda (rate)
    for ((RATE=0; RATE<${#ENTROPY_LAMBDA_LIST[@]}; RATE++));
    do
        # Loop over each GOP segment
        for ((GOP_ID=0; GOP_ID < $(((TOTAL_FRAME + GOP - 1)/GOP)) ; GOP_ID++));
        do
            echo "Running $SCENE"
            # Set experiment name and output directory
            EXP_NAME=$RESULT_DIR/${SCENE}/GOP_$GOP_ID/r$RATE
            # Calculate the starting frame for this GOP
            GOP_START_FRAME=$((FIRST_FRAME + GOP_ID * GOP ))
            # Calculate the maximum number of frames for this GOP
            MAX_GOP=$((TOTAL_FRAME + FIRST_FRAME - GOP_START_FRAME))
            # If this is the first GOP, train from scratch
            CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer_GIFStream.py $TYPE --disable_viewer --data_factor $DATA_FACTOR \
                --render_traj_path $RENDER_TRAJ_PATH --data_dir $SCENE_DIR/$SCENE/ --result_dir $EXP_NAME \
                --eval_steps 7000 30000 --save_steps 7000 30000 \
                --compression_sim --rd_lambda ${ENTROPY_LAMBDA_LIST[RATE]} --entropy_model_opt --rate $RATE \
                --batch_size 1 --GOP_size $(( MAX_GOP < GOP ? MAX_GOP : GOP)) --knn --start_frame $GOP_START_FRAME
        done
    done
done

# Run the summary script to aggregate results
python examples/summary.py --root_dir $RESULT_DIR
