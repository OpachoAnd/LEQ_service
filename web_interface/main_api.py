import json
import os
from pathlib import Path

import pika
from email_validator import validate_email, EmailNotValidError
from flask import Flask, request, redirect, flash, render_template
from werkzeug.utils import secure_filename

HOME_DIRECTORY = str(Path(__file__).absolute().parent)
UPLOAD_FOLDER = os.path.join(HOME_DIRECTORY, 'cloud')  # TODO, 'video')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = b'_5#y2L"A4Q8z\n\xec]/'

RMQ_HOST = 'mq'
RMQ_PORT = 5672
RMQ_USER = 'guest'
RMQ_PASSWORD = 'guest'

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
            # flash(str(e))
            print(str(e), flush=True)
            return redirect(request.url)

        if file_video.filename == '' or file_audio == '':
            # flash('No selected file')
            print('No selected file', flush=True)
            return redirect(request.url)

        if file_video and file_audio and emotion and e_mail:
            filename_video = secure_filename(file_video.filename)
            filename_audio = secure_filename(file_audio.filename)
            file_video.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_video))
            file_audio.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_audio))

            data = {'video': filename_video,
                    'audio': filename_audio,
                    'emotion': emotion,
                    'e_mail': e_mail}
            data = json.dumps(data)
            try:
                connection = pika.BlockingConnection(pika.ConnectionParameters(host=RMQ_HOST,
                                                                               port=RMQ_PORT))

                channel = connection.channel()
                channel.queue_declare(queue='ad_nerf')
                channel.basic_publish(exchange='', routing_key='ad_nerf', body=data)
                print('ok_connection', flush=True)
            except:
                print('error_connection', flush=True)

    return render_template('load_data.html', error=None)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
