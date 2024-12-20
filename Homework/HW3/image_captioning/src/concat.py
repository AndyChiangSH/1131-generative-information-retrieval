import json
import logging
import os

def concat_descriptions(original_path, captioned_path, output_path):
    """Concatenate photo descriptions from two JSONL files"""
    # Read and store data from both files
    original_data = {}
    with open(original_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            original_data[data['photo_id']] = data
            
    captioned_data = {}
    with open(captioned_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            captioned_data[data['photo_id']] = data
    
    # Combine descriptions and write output
    with open(output_path, 'w') as f:
        for photo_id, orig_item in original_data.items():
            # Get caption from second file if available
            capt_desc = captioned_data.get(photo_id, {}).get('photo_description', '')
            
            # Create combined item preserving all metadata
            combined_item = orig_item.copy()
            combined_item['photo_description'] = f"{orig_item['photo_description']}. Caption: {capt_desc}".strip()
            
            # Write to output file
            json.dump(combined_item, f)
            f.write('\n')

if __name__ == "__main__":
    CONFIG = {
        "input_path": "image_captioning/dataset/base+blip-image-captioning-large_1/",
        "concat_path": "image_captioning/dataset/git-large-coco_1/",
        "output_path": "image_captioning/dataset/base+blip-image-captioning-large_1+git-large-coco_1/",
    }
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    logging.info("> Start concat!")
    logging.info(f"Config: {CONFIG}")
    
    # Create output directory if it doesn't exist
    os.makedirs(CONFIG["output_path"], exist_ok=True)
    
    # Process both train and test files
    logging.info(f"\n> Concat train files...")
    concat_descriptions(
        os.path.join(CONFIG["input_path"], "train.jsonl"),
        os.path.join(CONFIG["concat_path"], "train.jsonl"),
        os.path.join(CONFIG["output_path"], "train.jsonl")
    )
    logging.info(f"\n> Concat test files...")
    concat_descriptions(
        os.path.join(CONFIG["input_path"], "test_images.jsonl"),
        os.path.join(CONFIG["concat_path"], "test_images.jsonl"),
        os.path.join(CONFIG["output_path"], "test_images.jsonl")
    )
    
    # Copy test.jsonl to output directory
    logging.info("> Copy test.jsonl to output directory...")
    os.system(
        f"cp {os.path.join(CONFIG['input_path'], 'test.jsonl')} {os.path.join(CONFIG['output_path'], 'test.jsonl')}")

    logging.info("> End concat!")