from flask import Flask, render_template, Response, stream_with_context
import cv2
import time

import mysql.connector as c
import matplotlib.pyplot as plt  
import numpy as np 

import datetime
now = datetime.datetime.now()
# print (now.strftime("%Y/%m/%d %H:%M:%S"))
mydb = c.connect(
    host="localhost",
    user="root",
    passwd="12345678",
    database="test"
)
mycursor = mydb.cursor()   
app = Flask(__name__)
camera = cv2.VideoCapture(0)

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
file_name = 'labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)  # 255/2 (grey level/2)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

signal = 0


class CircularQueue:

    def __init__(self, maxSize):
        self.queue = [None for i in range(maxSize)]
        self.maxSize = maxSize
        self.head = 0
        self.tail = 0

    def enqueue(self, data):
        self.queue[self.tail] = data
        self.tail = (self.tail+1) % self.maxSize
        return

    def last(self):
        return self.queue[(self.tail-2) % self.maxSize]

    def last2(self):
        return self.queue[(self.tail-3) % self.maxSize]


size = 4
q = CircularQueue(int(size))


for i in range(size):
    q.enqueue(100000)


def generate_frames():
    while (True):
        # read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# def analysis():
#     mycursor.execute("select count(*) from vehicles where time>'2023-02-06 19:13:06'")
#     mydb.commit()
#     myres = mycursor.fetchall()
#     countofvehicles=[]
#     for i in mycursor:
#         countofvehicles.append(i[0])


def blindturn():
    cap = camera

    # Check if the video is opened correctly

    # if not cap.isOpened():
    #     cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #    raise IOError("Cannot open video")

    font_scale = 3
    font = cv2.FONT_HERSHEY_PLAIN

    while True:

        ret, frame = cap.read()
        ClassIndex, confidece, bbox = model.detect(frame, confThreshold=0.55)
        if (len(ClassIndex) != 0):
            count = 0
            for i in ClassIndex:
                if (i == 1):
                    break
                count = count + 1
                

            if (count < len(ClassIndex)):
                area = abs((bbox[count][2]-bbox[count][0])
                           * (bbox[count][3]-bbox[count][1]))
                q.enqueue(area)
                print(area, q.last(), q.last2())
                if (area > q.last() and area > q.last2()):
                    signal = 0
                    # print ("red")
                    x=now.strftime('%Y-%m-%d %H:%M:%S')
                    mycursor.execute('insert into test.vehicles(timeval, vehiclecount) values(%s, %s)', (x, 1))
                    mydb.commit()
                    print("successfully committed") 

                    yield "red"
                    time.sleep(2)
                else:
                    signal = 1
                    # print ("green")
                    yield "green"
                    time.sleep(1)
            # cv2.putText(frame, font, fontScale=font_scale,
            #             color=(0, 255, 0), thickness=3)

        # cv2.imshow('Object Detection Tutorial', frame)


def stream_template(template_name, **context):
    app.update_template_context(context)
    t = app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    rv.disable_buffering()
    return rv


@app.route('/blindturn', methods=['POST', 'GET'])
def index():
    darshan = blindturn()
    return Response(stream_with_context(stream_template('index.html', darshan=darshan)))


@app.route('/')
def home():
    return render_template('blindturn.html')

@app.route('/cameraoutput')
def cameraoutput():
    return render_template('cameraoutput.html')


@app.route('/outputsignal')
def outputsignal():
    # blindturn()
    # return Response("hi",mimetype="text")
    # return Response(blindturn(), mimetype='multipart/x-mixed-replace; boundary=frame')

    x = "some data you want to return"
    return x, 200, {'Content-Type': 'text/css; charset=utf-8'}


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
