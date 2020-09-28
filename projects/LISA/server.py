# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# import for model
import glob
import os
import time
import cv2
import tqdm
import uuid
import shutil
import io
import numpy as np
from detectron2.data.detection_utils import read_image
from LISA import add_lisa_config
from setModel import getModel
from werkzeug.utils import secure_filename
from PIL import Image

# import for server
from flask import Flask, render_template, request, Response, send_file, jsonify
from queue import Queue, Empty
import threading
import time

# flask server
app = Flask(__name__)

# limit input file size under 2MB
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 

# model loading
demo = getModel()

# request queue setting
requests_queue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1

# static variable
INPUT_DIR = './input'
RESULT_DIR = './result'
POSIBLE_FORMAT = ['image/jpeg', 'image/jpg', 'image/png']

# request handling
def handle_requests_by_batch():
    try:
        while True:
            requests_batch = []
            while not (len(requests_batch) >= BATCH_SIZE):
                try:
                    requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
                except Empty:
                    continue
                
            batch_outputs = []

            for request in requests_batch:
                batch_outputs.append(run(request["input"][0]))

            for request, output in zip(requests_batch, batch_outputs):
                request["output"] = output
                
    except Exception as e:
        while not requests_queue.empty():
            requests_queue.get()
        print(e)


# request processing
threading.Thread(target=handle_requests_by_batch).start()

def byte_to_image(image_byte):
    open_image = Image.open(io.BytesIO(image_byte))
    image = rgba_to_rgb(open_image)
    return image
    
def rgba_to_rgb(open_image):
    if open_image.mode == 'RGBA':
        image = Image.new('RGB', open_image.size, (255, 255, 255))
        image.paste(open_image, mask=open_image.split()[3])
    else:
        image = open_image.convert('RGB')
    return image

def remove_file(paths):
    for path in paths:
        shutil.rmtree(path)

def image_to_byte(image):
    byte_io = io.BytesIO()
    image.save(byte_io, "PNG")
    byte_io.seek(0)
    return byte_io

# run model
def run(image_file):
    try:
        # read image_file
        f_id = str(uuid.uuid4())
        f_name = secure_filename(image_file.filename)
        image_byte = image_file.read()
        image = byte_to_image(image_byte)

        # make input&result file folder
        paths = [os.path.join(INPUT_DIR, f_id), os.path.join(RESULT_DIR, f_id)]
        os.makedirs(paths[0], exist_ok=True)
        os.makedirs(paths[1], exist_ok=True)

        # save image to input folder
        in_filename = os.path.join(paths[0], f_name)
        image.save(in_filename, quality=100, subsampling=0)

        # run model
        img = read_image(in_filename, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)

        # save result to result folder
        out_filename = os.path.join(paths[1], os.path.basename(in_filename))
        visualized_output.save(out_filename)

        # convert result image to byte format
        image = Image.open(out_filename)
        byte_image = image_to_byte(image)

        return byte_image

    except Exception as e:
        print(e)
        return 500

    finally:
        # remove input&output folder
        if paths:
            remove_file(paths)
            paths.clear()

# routing
@app.route("/detection", methods=['POST'])
def detection():
    try:
        # only get one request at a time
        if requests_queue.qsize() > BATCH_SIZE:
            return jsonify({'message' : 'TooManyReqeusts'}), 429
    
        # check image format
        try:
            image_file = request.files['image']
        except Exception:
            return jsonify({'message' : 'Image size is larger than 2MB'}), 400
    
        if image_file.content_type not in POSIBLE_FORMAT:
            return jsonify({'message' : 'Only support jpg, jpeg, png'}), 400

        # put data to request_queue
        req = {'input' : [image_file]}
        requests_queue.put(req)

        # wait output
        while 'output' not in req:
            time.sleep(CHECK_INTERVAL)
        
        # send output
        byte_image = req['output']

        if byte_image == 500:
            return jsonify({'message': 'Error! An unknown error occurred on the server'}), 500
        
        result = send_file(byte_image, mimetype="image/png")
        
        return result
    
    except Exception as e:
        print(e)
        return jsonify({'message': 'Error! Unable to process request'}), 400

@app.route('/health')
def health():
    return "ok"

@app.route('/')
def main():
    return render_template('index.html')

if __name__ == "__main__":
    from waitress import serve
    serve(app, host='0.0.0.0', port=80)
    
    



    
    
