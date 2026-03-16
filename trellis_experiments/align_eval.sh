#!/bin/bash

# Define your Python script name
PYTHON_SCRIPT="your_script.py"

# Loop through all .png files in the current directory
for file in outputs/teapot_more_angles/unaligned/*.vtp; do
# for file in outputs/asteroid/viridis/unaligned/tev_iso_0.08200450018048287*.vtp; do

    # Check if files actually exist to avoid errors if the folder is empty
    [ -e "$file" ] || continue

    echo "Processing: $file"
    
    # Call the python script and pass the filename as an argument
    # python3 "$PYTHON_SCRIPT" "$file"
    # python align_1.py --vtp outputs/teapot_explore/teapot.vtp --vtp2 "$file" \
    # --seed 0 --tmax 10 --deg 45 --noise 0.0 > "$file".log # --plot

    python align_connectivity.py --vtp outputs/teapot_explore/teapot.vtp --vtp2 "$file" \
    --save_plots_dir outputs/teapot_more_angles/plots/ \
    --save_aligned_vtp outputs/teapot_more_angles/aligned/ > "$file".log

    # python align_connectivity.py --vtp ../TheTruthPaper/imdb/outputs/asteroid_isosurface/tev_iso_0.08200450018048287.vtp --vtp2 "$file" \
    # --save_plots_dir outputs/asteroid/viridis/plots/ \
    # --save_aligned_vtp outputs/asteroid/viridis/aligned/ > "$file".log
done

echo "Done!"