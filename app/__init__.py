from flask import Flask
# from app import app
app = Flask(__name__)
# app.config['UPLOAD FOLDER'] = 'app/upload_video'

if __name__ == "__main__":
    app.run(debug=True)

from app import route