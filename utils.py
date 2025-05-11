import cv2
import numpy as np
from PIL import Image
import io

def apply_blur(image, box, blur_strength=30):
    """Apply blur to a specific region of the image"""
    img_array = np.array(image)
    x1, y1, x2, y2 = map(int, box)
    
    # Extract the region
    region = img_array[y1:y2, x1:x2]
    
    # Apply blur
    blurred_region = cv2.GaussianBlur(region, (blur_strength, blur_strength), 0)
    
    # Replace the region
    img_array[y1:y2, x1:x2] = blurred_region
    
    return Image.fromarray(img_array)

def apply_square(image, box):
    """Apply a square mask to a specific region"""
    img_array = np.array(image)
    x1, y1, x2, y2 = map(int, box)
    
    # Create a black square
    img_array[y1:y2, x1:x2] = 0
    
    return Image.fromarray(img_array)

def delete_object(image, box):
    """Delete an object by replacing it with surrounding pixels"""
    img_array = np.array(image)
    x1, y1, x2, y2 = map(int, box)
    
    # Get the surrounding pixels
    surrounding = img_array[max(0, y1-5):min(img_array.shape[0], y2+5),
                          max(0, x1-5):min(img_array.shape[1], x2+5)]
    
    # Calculate the average color
    avg_color = np.mean(surrounding, axis=(0, 1))
    
    # Replace the region with the average color
    img_array[y1:y2, x1:x2] = avg_color
    
    return Image.fromarray(img_array)

def crop_to_object(image, box):
    """Crop the image to the selected object"""
    x1, y1, x2, y2 = map(int, box)
    return image.crop((x1, y1, x2, y2))

def blur_faces(image, face_boxes):
    """Blur all faces in the image"""
    img_array = np.array(image)
    
    for box in face_boxes:
        x1, y1, x2, y2 = map(int, box)
        region = img_array[y1:y2, x1:x2]
        blurred_region = cv2.GaussianBlur(region, (30, 30), 0)
        img_array[y1:y2, x1:x2] = blurred_region
    
    return Image.fromarray(img_array)

def blur_number_plates(image, plate_boxes):
    """Blur all number plates in the image"""
    img_array = np.array(image)
    
    for box in plate_boxes:
        x1, y1, x2, y2 = map(int, box)
        region = img_array[y1:y2, x1:x2]
        blurred_region = cv2.GaussianBlur(region, (30, 30), 0)
        img_array[y1:y2, x1:x2] = blurred_region
    
    return Image.fromarray(img_array)

def prepare_download(image, high_res=False):
    """Prepare image for download"""
    if high_res:
        # Increase resolution by 2x
        width, height = image.size
        image = image.resize((width*2, height*2), Image.Resampling.LANCZOS)
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr 