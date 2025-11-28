import os
import argparse
from PIL import Image

# Register HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("⚠️  Warning: 'pillow-heif' not installed. HEIC support will be unavailable.")

def convert_to_jpg_inplace(folder, allowed_extensions=None):
    if allowed_extensions is None:
        allowed_extensions = ['.png', '.jpeg', '.bmp', '.gif', '.webp', '.heic', '.jpg']

    allowed_extensions = [ext.lower() for ext in allowed_extensions]
    all_files = os.listdir(folder)

    images = [
        f for f in all_files
        if os.path.splitext(f)[1].lower() in allowed_extensions and os.path.isfile(os.path.join(folder, f))
    ]
    images.sort()

    for filename in images:
        file_path = os.path.join(folder, filename)
        base, ext = os.path.splitext(filename)
        ext = ext.lower()

        # Skip if already .jpg
        if ext == '.jpg':
            continue

        new_filename = base + '.jpg'
        new_file_path = os.path.join(folder, new_filename)

        try:
            with Image.open(file_path) as img:
                rgb = img.convert('RGB')
                rgb.save(new_file_path, 'JPEG')
            os.remove(file_path)
            print(f"Converted {filename} → {new_filename}")
        except Exception as e:
            print(f"❌ Failed to convert {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Convert all images in a folder to JPG format in-place.')
    parser.add_argument('folder', type=str, help='Path to the folder with images.')
    parser.add_argument('--extensions', nargs='*', default=None, help='Optional list of extensions to include (e.g. .png .heic).')

    args = parser.parse_args()
    convert_to_jpg_inplace(args.folder, args.extensions)

if __name__ == '__main__':
    main()
