import os
import subprocess
import sys
root_dir = '/Users/bujack/Downloads/cube/code/output'
script = '/Users/bujack/Documents/nerf/code/reconstruct.py'
input_file = '/Users/bujack/Documents/nerf/data/cube.vti'

results = []

counter = 0
for name in os.listdir(root_dir):
    subdir = os.path.join(root_dir, name)
#    if counter > 20:
#        break
    if os.path.isdir(subdir) and 'checkpoint' in name:
        counter +=1

        print('subdir', subdir)
#        sys.exit()
        try:
            output = subprocess.check_output(
                ['python3.11', script, input_file, subdir],
                universal_newlines=True
            )
            for line in output.strip().splitlines():
                results.append(f'{subdir} {line}')
        except subprocess.CalledProcessError as e:
            print(f"Error in {subdir}: {e}")
print(results)
# Sort by last numeric value
def sort_key(line):
    try:
        return float(line.strip().split()[-1])
    except:
        return float('inf')

results.sort(key=sort_key)

with open('sorted_output.txt', 'w') as f:
    f.write('\n'.join(results))
