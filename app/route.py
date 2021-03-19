import tensorflow as tf
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pathlib
from tensorflow import keras
import numpy as np
import cv2
from app import app
import os
from flask import flash, render_template, redirect, request, Response, send_file , jsonify
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['mp4', 'wmv'])
upload_folder = 'F:/workkk/SeniorProject-Backend/app/upload_video'
models = tf.keras.models.load_model('F:/workkk/SeniorProject-Backend/app/model')
# Predictions = []

@app.route("/")
def upload():
    return render_template('upload.html',predictions = Predictions,grad_img=None)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/get_img', methods=['GET'])
def display():
    return send_file(grad_path, mimetype='image/jpg')

@app.route('/grad', methods=['GET'])
def sendfile():
    grad_path = "F:/workkk/SeniorProject-Backend/app/grad_cam/grad.jpg"
    return send_file(grad_path, mimetype='image/jpg')

@app.route('/upload', methods=['POST'])
# def reloadpage():
#     return render_template('upload.html',predictions = Predictions)
def upload_file():
    import numpy as np
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            status_code = Response(status=204)
            return status_code
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(upload_folder, filename))
            status_code = Response(status=200)

            Time_per_frame, video_path, frame_path = extract_frame()
            
            Result_per_frame = predict(models)
            # for i in range(3):
            #     Result_per_frame[i] = "normal"
            # print(Result_per_frame)
            smoke_frame_number = checkSmoke(Result_per_frame)
            # print(smoke_frame_number)
            if(smoke_frame_number == -1):
                remove_video(video_path)
                remove_frame(frame_path)
                data = {
                    'time' : '[None,]',
                    'gradcam': 'None',
                    'status_code' : '200'
                }
                return jsonify(data)
            else:
                grad_path = grad_cam(models, smoke_frame_number)
                remove_video(video_path)
                remove_frame(frame_path)
                # Predictions = str(Result_per_frame[smoke_frame_number:])
                Time = str(Time_per_frame[smoke_frame_number:])
                print(Time)
                data = {
                    'time'  : Time,
                    'gradcam' : grad_path.decode(),
                    'status_code' : '200'
                }
                return jsonify(data)
        else:
            status_code = Response(status=406)
            return status_code


def extract_frame():
    import numpy as np
    source = "F:/workkk/SeniorProject-Backend/app/upload_video/"
    video = os.listdir(source)
    print(source)
    cap = cv2.VideoCapture(source+video[0])
    total_frames = cap.get(7)
    print(total_frames)
    gap_time = total_frames/14
    print(gap_time)
    success, image = cap.read()
    count = 0
    target_folder = "F:/workkk/SeniorProject-Backend/app/video_frame/"
    os.chdir(target_folder)
    time = []
    while success:
        if(count > 0):
            cap.set(cv2.CAP_PROP_POS_MSEC, (count*gap_time*35))
            cv2.imwrite("frame%d.jpg" % count, image)
            time.append( round ( (count*gap_time*35) * 0.001 ,2) )
            # print((count*gap_time*35) * 0.001)
            success, image = cap.read()
            print('Read a new frame: ', success)
        count += 1
    os.chdir("F:/workkk/SeniorProject-Backend/app/")
    # os.remove(source+video[0])
    # print("Video Removed!")
    return time, source+video[0], target_folder

def remove_video(path):
    os.remove(path)
    print("Video Removed!")

def remove_frame(path):
    frame = os.listdir(path)
    for i in range(len(frame)):
        os.remove(path+frame[i])
    print("Frame Removed!")

def predict(models):
    model = models
    source = "F:/workkk/SeniorProject-Backend/app/video_frame/"
    frame = os.listdir(source)
    result = []
    # print(frame)
    for i in range(len(frame)):
        img_path = source + frame[i]
        img = keras.preprocessing.image.load_img(
            img_path, target_size=(224, 224))
        input_arr = keras.preprocessing.image.img_to_array(img)
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        input_arr = input_arr/255
        predictions = model.predict(input_arr)
        # print(predictions)
        if(predictions[0][0] > predictions[0][1]):
            result.append("normal")
        elif(predictions[0][0] < predictions[0][1]):
            result.append("smoke")
    return result

def checkSmoke(Predict):
    for i in range(len(Predict)):
        if Predict[i] == "smoke":
            return i
    return -1

def grad_cam(models, frame_number):
    import base64
    model = models
    source = "F:/workkk/SeniorProject-Backend/app/video_frame/"
    frame = os.listdir(source)

    def make_gradcam_heatmap(
        img_array, model, last_conv_layer_name, classifier_layer_names
    ):
        last_conv_layer = model.layers[1].get_output_at(-1)
        last_conv_layer_model = keras.Model(model.inputs, last_conv_layer)
        classifier_input = keras.Input(shape=last_conv_layer.shape[1:])
        x = classifier_input
        for layer_name in classifier_layer_names:
            x = model.get_layer(layer_name)(x)
        classifier_model = keras.Model(classifier_input, x)
        with tf.GradientTape() as tape:
            last_conv_layer_output = last_conv_layer_model(img_array)
            tape.watch(last_conv_layer_output)
            preds = classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]
        grads = tape.gradient(top_class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]
        heatmap = np.mean(last_conv_layer_output, axis=-1)
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        return heatmap
    last_conv_layer_name = "out_relu"
    classifier_layer_names = [
        "avg_pool",
        "batch_normalization",
        "top_dropout",
        "pred",
    ]
    img_path = source + frame[frame_number]
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    input_arr = keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    input_arr = input_arr/255
    img_array = input_arr
    heatmap = make_gradcam_heatmap(
        img_array, model, last_conv_layer_name, classifier_layer_names
    )

    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    save_path = "F:/workkk/SeniorProject-Backend/app/grad_cam/grad.jpg"
    superimposed_img.save(save_path)
    with open(save_path, "rb") as imageFile:
        str = base64.b64encode(imageFile.read())
    return str
