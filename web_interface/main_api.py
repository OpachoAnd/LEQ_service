import os
from pathlib import Path

from email_validator import validate_email, EmailNotValidError
from flask import Flask, request, redirect, flash, render_template
from werkzeug.utils import secure_filename


HOME_DIRECTORY = str(Path(__file__).absolute().parent.parent)
UPLOAD_FOLDER = os.path.join(HOME_DIRECTORY, 'cloud')  # TODO, 'video')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = b'_5#y2L"A4Q8z\n\xec]/'


# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'],
#                                filename)


@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')

        file_video = request.files['video']
        file_audio = request.files['audio']
        emotion = request.form.get(key='emotion')
        e_mail = request.form.get(key='email')

        try:
            validate_email(e_mail, check_deliverability=False)
        except EmailNotValidError as e:
            flash(str(e))
            return redirect(request.url)

        if file_video.filename == '' or file_audio == '':
            flash('No selected file')
            return redirect(request.url)

        if file_video and file_audio and emotion and e_mail:
            filename_video = secure_filename(file_video.filename)
            filename_audio = secure_filename(file_audio.filename)
            file_video.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_video))
            file_audio.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_audio))

    return render_template('load_data.html', error=None)


if __name__ == "__main__":
    # app.run(debug=True)
    print(UPLOAD_FOLDER)
