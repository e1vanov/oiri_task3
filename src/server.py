from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap

from flask import (Flask, 
                   request, 
                   url_for, 
                   send_file,
                   render_template, 
                   redirect, 
                   abort,
                   send_from_directory)

from flask_wtf.file import FileAllowed

from wtforms import (IntegerField, 
                     RadioField, 
                     SubmitField, 
                     FileField, 
                     FloatField, 
                     StringField)

from wtforms.validators import (InputRequired, 
                                DataRequired, 
                                NumberRange)

from class_image import Image
import numpy as np
import cv2
from skimage.morphology import skeletonize
import sknw

import os
import shutil


app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
Bootstrap(app)

images_list = []

class ImageForm(FlaskForm):

    image = FileField('Image', 
                      validators=[DataRequired('Specify file'),
                      FileAllowed(['jpg', 'jpeg'], 'Image formats only!')])

    submit = SubmitField('Submit')

@app.route('/')
@app.route('/index')
def index():
    try:
        return render_template('index.html')
    except:
        abort(400)

@app.route('/images')
def images():
    try:
        return render_template('images.html', images_ind=list(range(len(images_list))))
    except:
        abort(400)

@app.route('/topology/<int:ind>')
def topology(ind):
    try:
        return send_from_directory(images_list[ind].path, 'process.jpg')
    except:
        abort(400)

@app.route('/all_images/<int:ind>')
def all_images(ind):
    try:
        if images_list[ind].recognized == False:
            images_list[ind].recognize_circles()
        return render_template('image_page.html', 
                               ind=ind,
                               process_link=url_for('topology', ind=ind))
    except:
        abort(400)

@app.route('/new_image', methods=['GET','POST'])
def new_image():
    try:
        upload_image_form = ImageForm()

        if upload_image_form.validate_on_submit():

            image_extension = request.files['image'].filename.split('.')[-1]

            image_ind = len(images_list)
            path = 'tmp/images/' + str(image_ind)
            os.mkdir(path)
            upload_image_form.image.data.save(path + '/img.' + image_extension)

            new_image = Image(path, 
                              image_extension)
            images_list.append(new_image)

            return redirect(url_for('all_images', ind=len(images_list)-1))

        return render_template('from_form.html', form=upload_image_form)
    except:
        try:
            shutil.rmtree('tmp/images/' + str(image_ind))
        except FileNotFoundError:
            pass
        abort(400)

@app.errorhandler(400)
def handle_errors(e):
    return render_template('error_page.html')
