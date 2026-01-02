from pathlib import Path
from PIL import Image, ImageOps

def preprocess_with_padding(src_root, dest_root, min_images=99, target_size=(224, 224)):
    src_path = Path(src_root)
    dest_path = Path(dest_root)
    
    # Ensure the base processed directory exists
    dest_path.mkdir(parents=True, exist_ok=True)

    # Walk through each artist folder in data/raw/images
    for artist_dir in src_path.iterdir():
        if artist_dir.is_dir():
            # Create a list of valid image files in the subfolder
            images = [f for f in artist_dir.iterdir() if f.suffix.lower() in ['.jpg']]
            
            # STAGE 1: Check the 99-image threshold
            if len(images) >= min_images:
                print(f"Processing '{artist_dir.name}': {len(images)} images found.")
                
                # Create the specific artist folder only if they pass the threshold
                artist_dest = dest_path / artist_dir.name
                artist_dest.mkdir(exist_ok=True)
                
                # STAGE 2: Resize and Pad
                for img_path in images:
                    try:
                        with Image.open(img_path) as img:
                            # Convert to RGB to ensure consistency (removes Alpha channels/Grayscale issues)
                            img = img.convert('RGB')
                            
                            # ImageOps.pad resizes the image to fit within target_size 
                            # while maintaining aspect ratio, filling the rest with 'color'
                            res = ImageOps.pad(img, target_size, method=Image.Resampling.LANCZOS, color=(0, 0, 0))
                            
                            res.save(artist_dest / img_path.name)
                    except Exception as e:
                        print(f"Error processing {img_path.name}: {e}")
            else:
                print(f"Skipping '{artist_dir.name}': Only {len(images)} images")

if __name__ == "__main__":
    # Update these paths if your folder structure differs
    SOURCE = "data/raw/images"
    DESTINATION = "data/processed/resized224p"
    
    preprocess_with_padding(SOURCE, DESTINATION, min_images=99)
    print("Done! Check data/processed/resized224p for your images.")