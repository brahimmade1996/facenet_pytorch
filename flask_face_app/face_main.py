from facenet_pytorch import MTCNN
import torch
from PIL import Image
import time
from flask import Flask, render_template,request
import os
from PIL import Image, ImageDraw
app = Flask(__name__)
idx = 0

def box_check(boxes, img_size):
#boxes = [[box1],[box2]]
    box = True
    if len(boxes)== 1:
        box_w = boxes[0][2]-boxes[0][0]
        box_h = boxes[0][3]-boxes[0][1]
        box_wh = box_w * box_h
        img_w, img_h = img_size
        img_wh = img_w * img_h
        box_bottom = img_h - boxes[0][3]

        if box_wh/img_wh*100 > 15:
            if box_bottom < box_h:
                box = False
                print(int(box_wh/img_wh * 100),'%')


    elif len(boxes) >= 4:
        box = False
        print(len(boxes),'over!!!!!!')
    return box 
    #return render_template('result.html', box=box)
    #return render_template('result.html',box=box)

@app.route('/upload')
def index():
    return render_template('upload.html')
    
    
@app.route('/predict', methods = ['GET','POST'])
def predict():
    global idx 
    idx += 1
    if request.method == 'POST':
        f = request.files['file']
        f.save('input/'+str(idx)+'.jpg')
    
    #device check    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    start = time.time()
    mtcnn = MTCNN(keep_all=True, device=device, thresholds=[0.8,0.9,0.9])
    print(f.filename)
    img = Image.open('input/'+str(idx)+'.jpg')
    img_size = [img.size[0],img.size[1]]
    
    boxes, _ = mtcnn.detect(img)
    print(len(boxes),'num')
    
    frame_draw = img.copy()
    draw = ImageDraw.Draw(frame_draw)
    for box in boxes:
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=4)
    path = str(idx)+'.jpg'
    frame_draw.save('static/'+str(idx)+'.jpg')
    
    detect_face = box_check(boxes, img_size)
    
    end = time.time() - start
    print(end,'sec')

    #return str(detect_face)
    return render_template('result.html', box=detect_face, path = path)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port =8886)
