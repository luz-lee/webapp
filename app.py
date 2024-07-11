from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
from utils import apply_watermark, extract_watermark, apply_watermark_to_video, extract_watermark_from_video

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['EXTRACT_FOLDER'] = 'extracts'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Handle video upload
            video_file = request.files.get('video')
            image_file = request.files.get('image')
            text = request.form['text']
            watermark_image = request.files.get('watermark_image')
            use_image = False

            if video_file:
                if watermark_image:
                    watermark_filename = secure_filename(watermark_image.filename)
                    watermark_image_path = os.path.join(app.config['UPLOAD_FOLDER'], watermark_filename)
                    watermark_image.save(watermark_image_path)
                    use_image = True
                else:
                    watermark_image_path = text

                if text or use_image:
                    filename = secure_filename(video_file.filename)
                    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    video_file.save(input_path)

                    # Apply watermark
                    watermarked_filename = 'watermarked_' + filename
                    watermarked_path = os.path.join(app.config['RESULT_FOLDER'], watermarked_filename)
                    apply_watermark_to_video(input_path, watermarked_path, watermark_image_path, frame_skip=2, use_image=use_image)

                    # Extract watermark
                    extracted_filename = 'extracted_' + filename
                    extracted_path = os.path.join(app.config['EXTRACT_FOLDER'], extracted_filename)
                    extract_watermark_from_video(input_path, watermarked_path, extracted_path, frame_skip=2)

                    return render_template('index.html', original_url=filename, watermarked_url=watermarked_filename, extracted_url=extracted_filename)
            
            # Handle image upload
            if image_file and text:
                filename = secure_filename(image_file.filename)
                input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(input_path)

                # Apply watermark
                img = cv2.imread(input_path)
                watermarked_img = apply_watermark(img, text)

                watermarked_filename = 'watermarked_' + filename
                watermarked_path = os.path.join(app.config['RESULT_FOLDER'], watermarked_filename)
                cv2.imwrite(watermarked_path, watermarked_img)

                # Extract watermark
                extracted_filename = 'extracted_' + filename
                extracted_path = os.path.join(app.config['EXTRACT_FOLDER'], extracted_filename)
                watermark, result2 = extract_watermark(img, watermarked_img)
                cv2.imwrite(extracted_path, result2)

                return render_template('index.html', original_url=filename, watermarked_url=watermarked_filename, extracted_url=extracted_filename)
        except Exception as e:
            print(f"Error handling request: {e}")
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/extracts/<filename>')
def extract_file(filename):
    return send_from_directory(app.config['EXTRACT_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['RESULT_FOLDER']):
        os.makedirs(app.config['RESULT_FOLDER'])
    if not os.path.exists(app.config['EXTRACT_FOLDER']):
        os.makedirs(app.config['EXTRACT_FOLDER'])
    app.run(debug=True)
