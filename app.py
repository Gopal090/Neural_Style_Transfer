import os
import io
import time
import base64
import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired

# Import existing AdaIN models and utilities
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}
try:
    from flask_bootstrap import Bootstrap4 as Bootstrap
except ImportError:
    from flask_bootstrap import Bootstrap

Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_image = FileField('Content Image')
    style_image = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Initializing Neural Style Transfer on device: {device}")

# Load models
vgg_path = 'vgg_normalised.pth'
encoder = VGGEncoder(vgg_path).to(device)

decoder = Decoder().to(device)
# Yesterday's big_dataset training checkpoint
decoder_path = os.path.join('experiment', 'big_dataset', 'decoder_final.pth')
if not os.path.exists(decoder_path):
    # Fallback if relative path differs
    decoder_path = r'experiment/big_dataset/decoder_final.pth'

print(f"[*] Loading decoder checkpoint from: {decoder_path}")
decoder.load_state_dict(torch.load(decoder_path, map_location=device))

encoder.eval()
decoder.eval()
print("[*] VGG-19 Encoder and AdaIN Decoder successfully loaded and ready.")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def style_transfer(content_image, style_image, encoder, decoder, alpha=1.0, device=device, resolution=512, preserve_color=False):
    """
    Executes Arbitrary Neural Style Transfer using AdaIN with resolution & optional color preservation.
    """
    img_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor()
    ])

    c_tensor = img_transform(content_image).unsqueeze(0).to(device)
    s_tensor = img_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(c_tensor, is_test=True)
        style_feats = encoder(s_tensor, is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)

        # Style strength interpolation
        stylized_feats = alpha * stylized_feats + (1.0 - alpha) * content_feats

        stylized_tensor = decoder(stylized_feats)
        stylized_tensor = torch.clamp(stylized_tensor, 0.0, 1.0)

    # Convert to PIL Image
    output_img = transforms.ToPILImage()(stylized_tensor.squeeze(0).cpu())

    if preserve_color:
        # Match luminance of styled image with original content colors
        c_resized = content_image.resize((resolution, resolution))
        c_ycbcr = c_resized.convert('YCbCr')
        o_ycbcr = output_img.convert('YCbCr')
        _, cb, cr = c_ycbcr.split()
        y, _, _ = o_ycbcr.split()
        output_img = Image.merge('YCbCr', (y, cb, cr)).convert('RGB')

    return output_img


def save_image(pil_image, path):
    pil_image.save(path, format='JPEG', quality=95)


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if request.method == 'POST':
        # Check standard form submission
        if form.validate_on_submit():
            # Handle content upload
            c_file = form.content.data or form.content_image.data
            if c_file and getattr(c_file, 'filename', ''):
                if allowed_file(c_file.filename):
                    content_filename = secure_filename(c_file.filename)
                    c_file.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                    form.content_path.data = content_filename
            else:
                content_filename = form.content_path.data

            # Handle style upload
            s_file = form.style.data or form.style_image.data
            if s_file and getattr(s_file, 'filename', ''):
                if allowed_file(s_file.filename):
                    style_filename = secure_filename(s_file.filename)
                    s_file.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                    form.style_path.data = style_filename
            else:
                style_filename = form.style_path.data

            if content_filename and style_filename:
                content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
                style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)

                try:
                    content_image = Image.open(content_path).convert('RGB')
                    style_image = Image.open(style_path).convert('RGB')

                    alpha = float(form.alpha.data) if form.alpha.data is not None else 1.0
                    output_image = style_transfer(content_image, style_image, encoder, decoder, alpha=alpha, device=device)

                    # Timestamped unique filename to prevent browser cache collision
                    timestamp = int(time.time())
                    result_filename = f"stylized_{timestamp}_{content_filename}"
                    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                    save_image(output_image, result_path)

                    result_image = result_filename
                except Exception as e:
                    error = f"Style transfer failed: {str(e)}"
            else:
                if not content_filename:
                    error = 'Please upload a content image.'
                elif not style_filename:
                    error = 'Please upload a style image.'
        else:
            if not form.content.data and not form.content_path.data:
                error = 'Please upload a content image.'
            elif not form.style.data and not form.style_path.data:
                error = 'Please upload a style image.'

    return render_template('index.html', form=form, result_image=result_image,
                           content_image=content_filename, style_image=style_filename, error=error)


@app.route('/stylize', methods=['POST'])
@app.route('/transfer', methods=['POST'])
def api_stylize():
    """
    API endpoint for interactive AJAX requests from index.html (with real-time progress & comparison slider).
    """
    start_time = time.time()

    content_file = request.files.get('content_image') or request.files.get('content')
    style_file = request.files.get('style_image') or request.files.get('style')

    if not content_file or not style_file:
        return jsonify({'error': 'Please provide both content and style image files.'}), 400

    if content_file.filename == '' or style_file.filename == '':
        return jsonify({'error': 'One of the uploaded files is empty.'}), 400

    try:
        content_img = Image.open(content_file.stream).convert('RGB')
        style_img = Image.open(style_file.stream).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Failed to decode uploaded image: {str(e)}'}), 400

    alpha = float(request.form.get('alpha', 1.0))
    resolution = int(request.form.get('resolution', 512))
    preserve_color = request.form.get('preserve_color') in ['true', 'True', '1', True]

    try:
        output_img = style_transfer(
            content_img, style_img, encoder, decoder,
            alpha=alpha, device=device, resolution=resolution, preserve_color=preserve_color
        )

        buffered = io.BytesIO()
        output_img.save(buffered, format="JPEG", quality=95)
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        elapsed = time.time() - start_time

        return jsonify({
            'success': True,
            'result_image': f'data:image/jpeg;base64,{img_b64}',
            'time_taken': f'{elapsed:.2f}s',
            'resolution': f'{resolution}x{resolution}',
            'alpha': alpha
        })
    except Exception as e:
        return jsonify({'error': f'Style transfer processing failed: {str(e)}'}), 500


@app.route('/uploads/<path:filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)


if __name__ == '__main__':
    print("[*] Starting Neural Style Transfer Server at http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)
