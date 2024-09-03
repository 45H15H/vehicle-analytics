import os
import csv
import json
from PIL import Image
import google.generativeai as genai
import time

# Configure the Generative AI model
genai.configure(api_key="AIzaSyAh2OjBYiM5rsIc-rGiTUSQuBl-xAeYWpA")

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

prompt_template = """Return analysis of the car. Include the color, make, type, and license plate. Return result in the following format like a python dictionary: {"color": "red", "make": "Toyota", "type": "car", "license plate": "ABC123"}."""

def analyze_image(image_path):
    img = Image.open(image_path)
    response = model.generate_content([prompt_template, img])
    try:
        dictionary = json.loads(response.text)
        return dictionary
    except json.JSONDecodeError:
        print(f"Error decoding JSON for image {image_path}")
        return None

def process_images_in_folder(folder_path, output_csv):
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing {image_path}")
            result = analyze_image(image_path)
            if result:
                result['filename'] = filename
                results.append(result)
            time.sleep(1)

    # Save results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'color', 'make', 'type', 'license plate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

# Specify the folder containing images and the output CSV file
# folder_path = r'.\blurred_images'
folder_path = r'.\deblurred_images'
output_csv = 'vehicle_analytics_result.csv'

# Process images and save results to CSV
process_images_in_folder(folder_path, output_csv)
print(f"Results saved to {output_csv}")