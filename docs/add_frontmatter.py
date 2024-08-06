import os
import argparse


def add_frontmatter_to_markdown(file_path):
    file_name = os.path.basename(file_path)
    title = os.path.splitext(file_name)[0]

    frontmatter = f"""---
title: "{title}"
description: "NeuroStatX is a command-line toolbox to perform statistical
 analysis on neuroscience data."
---"""

    with open(file_path, 'r') as file:
        content = file.read()

    with open(file_path, 'w') as file:
        file.write(frontmatter + '\n' + content)


def process_markdown_files(file_paths):
    for file_path in file_paths:
        add_frontmatter_to_markdown(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Add frontmatter to Markdown files.')
    parser.add_argument('files', nargs='+', type=str,
                        help='List of Markdown files')

    args = parser.parse_args()
    process_markdown_files(args.files)
