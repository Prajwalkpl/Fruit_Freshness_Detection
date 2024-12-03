from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename


app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


model = load_model('aob.keras')
classes = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges','FreshGrape','FreshGuava','FreshJujube','FreshPomegranate','FreshStrawberry','RottenGrape','RottenGuava','RottenJujube','RottenPomegranate','RottenStrawberry']

def preprocess_image(image_path):
    """Preprocess uploaded image for prediction."""
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Rescale image
    return x

@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the home page and handle image uploads."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            img = preprocess_image(file_path)
            preds = model.predict(img)
            result = classes[np.argmax(preds)]
            result=result.lower()
    
            freshness = "Fresh" if 'fresh' in result else "Rotten"
            return render_template('result.html', result=result, freshness=freshness, filename=filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return url_for('static', filename='uploads/' + filename)

if __name__ == '__main__':
    app.run(debug=True)
