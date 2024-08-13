from flask import Flask, request, render_template, jsonify, send_from_directory
import os
from PIL import Image
import execute_function
from datetime import datetime

app = Flask(__name__)
app = Flask(__name__, static_url_path='/static')

# url to the 3DGS model
file_url = "http://127.0.0.1:5000/download/2024-06-04_18-47-10/output/1d55d614-3/point_cloud/iteration_3000/point_cloud.ply"

# Define the directory where uploaded files will be stored
UPLOAD_FOLDER = '../data'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define the directory where generated 3DGS models are stored
MODEL_FOLDER = '../data'

# Route to render the HTML form for uploading images
@app.route('/')
def upload_form():
    model_files = list_model_files(MODEL_FOLDER)
    return render_template('index.html', model_files=model_files)

def resize_image(image, max_resolution=980):
    width, height = image.size
    if width > max_resolution or height > max_resolution:
        if width > height:
            new_width = max_resolution
            new_height = int((max_resolution / width) * height)
        else:
            new_height = max_resolution
            new_width = int((max_resolution / height) * width)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    return image

def convert_to_jpg(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def list_model_files(model_folder):
    model_files = []
    valid_folders = {'iteration_3000', 'iteration_7000', 'iteration_10000', 'iteration_30000'}
    
    for root, dirs, files in os.walk(model_folder):
        for file in files:
            if file.endswith('.ply'):
                relative_path = os.path.relpath(os.path.join(root, file), model_folder)
                # Get the folder name immediately under the model_folder
                folder_name = os.path.basename(os.path.dirname(relative_path))
                if folder_name in valid_folders:
                    model_files.append(relative_path)
    
    return model_files

# Route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_files():
    # Create a folder with the current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    upload_subfolder = os.path.join(UPLOAD_FOLDER, current_datetime)
    os.makedirs(upload_subfolder, exist_ok=True)

    # Create a folder for images inside the current date and time folder
    images_folder = os.path.join(upload_subfolder, 'images')
    os.makedirs(images_folder, exist_ok=True)

    # Check if image files are uploaded
    if 'images[]' in request.files:
        images = request.files.getlist('images[]')
        for image in images:
            if image.filename == '':
                return 'No selected image file'
            
            # Open the uploaded image file
            img = Image.open(image)
            
            # Resize the image if necessary
            img = resize_image(img)
            
            # Convert to JPEG if necessary
            if image.filename.split('.')[-1].lower() not in ['jpeg', 'jpg']:
                img = convert_to_jpg(img)
            
            # Save the image
            image_path = os.path.join(images_folder, os.path.splitext(image.filename)[0] + '.jpg')
            img.save(image_path, format='JPEG')


    # Execute COLMAP and SAM function
    colmap_output = execute_function.execute_colmap_in_container(current_datetime)
    sam_output = execute_function.execute_sam_in_container(current_datetime)
    cgc_output = execute_function.execute_cgc_in_container(current_datetime)

    return jsonify({'message': 'Files uploaded and processed successfully', 'colmap_output': colmap_output, 'sam_output': sam_output, 'cgc_output': cgc_output})


@app.route('/3dgs_viewer')
def view_3dgs():
    model_files = list_model_files(MODEL_FOLDER)
    return render_template('3dgs_viewer/index.html', model_files=model_files)

@app.route('/download/<path:filename>')
def download_file(filename):
    data_directory = '/app/data/'  
    return send_from_directory(data_directory, filename, as_attachment=True)

# API endpoint to get the URL value for the 3D file
@app.route('/get_3d_file_url')
def get_3d_file_url():
    return jsonify({'file_url': file_url})

# Route to set the file URL when a link is clicked
@app.route('/set_3d_file_url', methods=['POST'])
def set_3d_file_url():
    global file_url
    file_url = "http://127.0.0.1:5000/download/" + request.form['url']
    return 'File URL set successfully'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
