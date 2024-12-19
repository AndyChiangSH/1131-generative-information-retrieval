import json
import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import logging
from tqdm import tqdm


def image_captioning(input_path, image_dir, output_dir, model_name):    
    # Load image captioning model and processor
    logging.info(">> Load image captioning model and processor...")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f">> Device: {device}")
    
    # Move model to GPU if available
    model = model.to(device)
    
    # Read input JSONL file
    logging.info(">> Read input JSONL file...")
    processed_data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            # Parse JSON line
            data = json.loads(line)
            
            # Load and process image
            image_path = os.path.join(image_dir, data['photo_path'])
            try:
                raw_image = Image.open(image_path).convert('RGB')

                # conditional image captioning
                text = ""
                inputs = processor(raw_image, text, return_tensors="pt").to(device)

                out = model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)
                
                # Update photo description
                data['photo_description'] = caption
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                
            processed_data.append(data)
    
    # Write processed data to new JSONL file
    logging.info(">> Write processed data to new JSONL file...")
    with open(output_dir, 'w', encoding='utf-8') as f:
        for data in processed_data:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')

    
if __name__ == '__main__':
    CONFIG = {
        "model_name": "Salesforce/blip-image-captioning-large",
        "input_path": "dataset/",
        "output_path": "image_captioning/dataset/blip-image-captioning-large_2/",
        "log_path": "image_captioning/log/blip-image-captioning-large_2.log",
    }
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(CONFIG["log_path"]),
            logging.StreamHandler()
        ]
    )
    
    # Log config and start
    logging.info("> Start image captioning!")
    logging.info(f"Config: {CONFIG}")
    
    # Create output directory if it doesn't exist
    os.makedirs(CONFIG["output_path"], exist_ok=True)

    # Run image captioning
    logging.info("> Image captioning for train data...")
    image_captioning(
        os.path.join(CONFIG["input_path"], "train.jsonl"),
        os.path.join(CONFIG["input_path"], "train_images/train_images/"),
        os.path.join(CONFIG["output_path"], "train.jsonl"),
        CONFIG["model_name"]
    )
    logging.info("> Image captioning for test data...")
    image_captioning(
        os.path.join(CONFIG["input_path"], "test_images.jsonl"),
        os.path.join(CONFIG["input_path"], "test_images/test_images/"),
        os.path.join(CONFIG["output_path"], "test_images.jsonl"),
        CONFIG["model_name"]
    )
    
    # Copy test.jsonl to output directory
    logging.info("> Copy test.jsonl to output directory...")
    os.system(f"cp {os.path.join(CONFIG['input_path'], 'test.jsonl')} {os.path.join(CONFIG['output_path'], 'test.jsonl')}")
    
    logging.info("> End image captioning!")
