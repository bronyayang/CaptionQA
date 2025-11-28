import os
import argparse
import json

def get_next_mapping_filename(source_dir):
    base_filename = 'mapping'
    ext = '.json'
    version = 1
    mapping_file = os.path.join(source_dir, f'{base_filename}_{version}{ext}')
    # If the base mapping file exists, find the next available version number.
    while os.path.exists(mapping_file):
        version += 1
        mapping_file = os.path.join(source_dir, f'{base_filename}_{version}{ext}')
    return mapping_file

def rename_images_inplace(source_dir, image_extensions=None):
    # Default to common image extensions in lowercase if none provided.
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    else:
        image_extensions = [ext.lower() for ext in image_extensions]
    
    # List and filter files by image extensions.
    all_files = os.listdir(source_dir)
    images = [
        f for f in all_files
        if os.path.splitext(f)[1].lower() in image_extensions and os.path.isfile(os.path.join(source_dir, f))
    ]
    images.sort()

    mapping = {}
    
    # Rename files in place and build mapping dictionary.
    for i, filename in enumerate(images, start=1):
        ext = os.path.splitext(filename)[1].lower()
        new_name = f'{i}{ext}'
        src_file = os.path.join(source_dir, filename)
        dest_file = os.path.join(source_dir, new_name)
        os.rename(src_file, dest_file)
        mapping[filename] = new_name
        print(f'Renamed: {filename} -> {new_name}')

    # Determine the next available mapping filename.
    mapping_file = get_next_mapping_filename(source_dir)
    
    # Save mapping as JSON.
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=4)
    
    print(f'\nMapping saved to {mapping_file}')

def main():
    parser = argparse.ArgumentParser(
        description='Rename image files in place and save a JSON mapping of original to new filenames.'
    )
    parser.add_argument(
        'source_dir', 
        type=str,
        help='The source directory containing the images to rename.'
    )
    parser.add_argument(
        '--image_extensions',
        nargs='*',
        default=None,
        help='Optional list of image file extensions (e.g., .jpg .png).'
    )

    args = parser.parse_args()
    rename_images_inplace(args.source_dir, args.image_extensions)

if __name__ == '__main__':
    main()
