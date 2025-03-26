import os
import shutil
import re

source_dir = 'preprocess'
target_dir = 'preprocess_split'

# Create target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Get all files in source directory
files = os.listdir(source_dir)

for file in files:
    # Check if file starts with 'Area_' followed by a digit
    match = re.match(r'Area_([1-5])', file)
    if match:
        digit = match.group(1)
        if digit == '5':
            # Rename to start with 'test' and copy to target directory
            new_file = 'test_' + file
        else:  # digit in ['1', '2', '3', '4']
            # Rename to start with 'train' and copy to target directory
            new_file = 'train_' + file
        shutil.copy(os.path.join(source_dir, file), os.path.join(target_dir, new_file))