#!/bin/bash

# Set variables
REFERENCES_RELATIVE_PATH=".context/references"  # Relative path of references directory to project root
REFERENCES_DIR=$(realpath "$(dirname "$0")/../$REFERENCES_RELATIVE_PATH")
REFERENCES_LIST="$REFERENCES_DIR/references-list.txt"
GITIGNORE_FILE=$(realpath "$(dirname "$0")/../.gitignore")

# Check if references directory exists, create if it doesn't
if [ ! -d "$REFERENCES_DIR" ]; then
  echo "Creating .references directory..."
  mkdir -p "$REFERENCES_DIR"
fi

# Check if references-list.txt exists
if [ ! -f "$REFERENCES_LIST" ]; then
  echo "Error: $REFERENCES_LIST does not exist!"
  echo "Please create this file and add GitHub repository links, one per line."
  exit 1
fi

echo "Starting to process reference projects..."

# Read each line from references-list.txt file
while IFS= read -r repo_url || [[ -n "$repo_url" ]]; do
  # Skip empty lines and comment lines
  if [[ -z "$repo_url" || "$repo_url" =~ ^# ]]; then
    continue
  fi
  
  echo "Processing repository: $repo_url"
  
  # Extract repository name from URL
  repo_name=$(basename "$repo_url" .git)
  target_dir="$REFERENCES_DIR/$repo_name"
  
  # Check if directory name is the path to be added to .gitignore
  gitignore_path="$REFERENCES_RELATIVE_PATH/$repo_name"
  
  # Check if already in .gitignore
  if ! grep -q "^$gitignore_path\$" "$GITIGNORE_FILE"; then
    echo "Adding $gitignore_path to .gitignore..."
    # Ensure the added entry is on a new line
    # First check if the file ends with a newline character
    if [ -f "$GITIGNORE_FILE" ] && [ -s "$GITIGNORE_FILE" ]; then
      last_char=$(tail -c 1 "$GITIGNORE_FILE")
      if [ "$last_char" != "" ]; then
        # If the last line is not empty, add a newline character first
        echo "" >> "$GITIGNORE_FILE"
      fi
    fi
    # Add ignore path
    echo "$gitignore_path" >> "$GITIGNORE_FILE"
  else
    echo "$gitignore_path is already in .gitignore"
  fi
  
  # Check if directory exists
  if [ -d "$target_dir" ]; then
    # Directory exists, try to update
    echo "Updating $repo_name..."
    cd "$target_dir" && git pull && cd - > /dev/null
  else
    # Directory doesn't exist, clone repository
    echo "Cloning $repo_name..."
    git clone "$repo_url" "$target_dir"
  fi
  
done < "$REFERENCES_LIST"

echo "All reference projects processed!"