source config.txt

if [ "$thres_option" == "list" ]; then
    threshold=("${thres_list[@]}")
elif [ "$thres_option" == "spec" ]; then
    threshold=("$thres")
else
    echo "error: list or spec"
fi

if [[ "$debug_mode" == "y" || "$debug_mode" == "Y" ]]
then
    debug_option="--debug"
    p1_option="--p1 $p1_input"
    p2_option="--p2 $p2_input"
else
    debug_option=""
    p1_option=""
    p2_option=""
fi

for thres in "${threshold[@]}"; do
    if [[ "$split_mode" == "awmr" || "$split_mode" == "AWMR" ]]
    then
        python 2-b_arg_TSDF_awmr.py "$dataset_name" "$finest_voxel_size" "$thres" "$sample_option" $debug_option $p1_option $p2_option --version 1
        echo "Split and Mesh done, [dataset_name: $dataset_name], [thres: $thres], [split_mode: $split_mode]."
    else
        python 2-c_arg_TSDF_octree.py "$dataset_name" "$finest_voxel_size" "$thres" "$sample_option" $debug_option $p1_option $p2_option --version 1
        echo "Split and Mesh done, [dataset_name: $dataset_name], [thres: $thres], [split_mode: $split_mode]."
    fi
done