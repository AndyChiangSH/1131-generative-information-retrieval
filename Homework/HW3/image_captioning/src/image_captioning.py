import json
import os
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import logging
from tqdm import tqdm


def image_captioning(input_path, image_dir, output_dir, model_name):    
    # Load image captioning model and processor
    logging.info(">> Load image captioning model and processor...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(model_name)
    
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
                # Open image
                image = Image.open(image_path)
                
                # Process image
                inputs = processor(images=image, return_tensors="pt")
                # Move inputs to same device as model
                inputs = {k: v.to(device) for k, v in inputs.items()}
                # Generate caption
                outputs = model.generate(
                    **inputs,  # Pass all inputs directly
                    max_length=50,
                    num_return_sequences=1
                )
                
                # Move outputs back to CPU for decoding
                outputs = outputs.cpu()
                caption = processor.decode(outputs[0], skip_special_tokens=True)
                
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
        "model_name": "microsoft/git-base",
        "input_path": "dataset/",
        "output_path": "image_captioning/dataset/git-base_1/",
        "log_path": "image_captioning/log/git-base_1.log",
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
