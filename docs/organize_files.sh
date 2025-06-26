#!/bin/bash

# Check if the output directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <output_directory>"
  exit 1
fi

# Set the output directory
output_dir=$1

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Iterate over all markdown files
for file in neurostatx.*.*.md; do
  # Extract the middle part and third part of the filename
  middle_part=$(echo "$file" | cut -d'.' -f2)
  third_part=$(echo "$file" | cut -d'.' -f3)
  
  # Convert middle part to uppercase (optional, if you want the folders to be in uppercase)
  folder_name=$(echo "$middle_part")

  # Modify the file content to convert the specified lines
  sed -i -E "s/(neurostatx\.$middle_part\.)$third_part\.(.*)/ \2/" "$file"
  
  # Replace all # with ####
  sed -i 's/^# /#### /' "$file"

  # Replace title in the frontmatter
  sed -i "s/title: \"neurostatx\.$middle_part\.$third_part\"/title: \"$middle_part\.$third_part\"/" "$file"  

  # Create the folder within the output directory if it doesn't exist
  mkdir -p "$output_dir/$folder_name"
  
  # Move and rename the file into the folder within the output directory
  mv "$file" "$output_dir/$folder_name/$third_part.md"  

done
