#!/bin/bash

# Define your Python script name
PYTHON_SCRIPT="your_script.py"

# Loop through all .png files in the current directory
for file in outputs/teapot_more_angles/*.glb; do
    # Check if files actually exist to avoid errors if the folder is empty
    [ -e "$file" ] || continue

    echo "Processing: $file"
    
    # Call the python script and pass the filename as an argument
    # python3 "$PYTHON_SCRIPT" "$file"
    # python align_1.py --vtp outputs/teapot_explore/teapot.vtp --vtp2 "$file" \
    # --seed 0 --tmax 10 --deg 45 --noise 0.0 > "$file".log # --plot
    stem="${file%.*}"

    python  convertGlb2Vtk.py "$file" --out "$stem".vtp
done

echo "Done!"