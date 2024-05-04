import os
import cv2
import time
import base64
import threading
from ultralytics import YOLO
from datetime import datetime
from multiprocessing import Process
# from paho.mqtt import client as mqtt_client
# import paho
import paho.mqtt.client as mqtt_client
from paho import mqtt

model = YOLO("yolov5nu.pt")
host = '*'
port = 8883
username = "*"
password = "*"
threshold = 0.8  # Ngưỡng chấp nhận để gửi ảnh (%)

# Hàm gửi ảnh (lưu vào local và gửi qua mqtt)
def send_image(box, frame, now):
    x1, y1, x2, y2 = box.xyxy[0]  # lưu thông số kích thước của box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    crop_person = frame[y1:y2, x1:x2]  # cắt phần ảnh có người
    id_person = str(int(box.id[0]))
    year, month, day, hour, minute, second = (
        str(now.year),
        str(now.month),
        str(now.day),
        str(now.hour),
        str(now.minute),
        str(now.second),
    )
    file_path = ".\\detected"  # Tạo thư mục detect
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    file_path += "\\" + day + "-" + month + "-" + year  # mỗi ngày là một thư mực
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    file_path += "\\" + "ID" + id_person  # thư mục mỗi ngày chứa các thư mục ID người
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    # cái này để lưu ảnh vào local
    image_name = (
        file_path + "\\"
        + hour + "-"
        + minute + "-"
        + second + ".jpg"
    )
    is_saved = cv2.imwrite(image_name, crop_person)
    if is_saved:
        print("ảnh đã được lưu")
    else:
        print("ERROR")
    client = connect_mqtt()
    client.loop_start()
    topic = ("detected" + "/" 
            + day + "-" + month + "-" + year + "/image/"
            + "ID" + id_person)
    file = open(image_name,"rb")
    img_bytes = file.read()
    image_b64 = base64.b64encode(img_bytes).decode("utf-8")
    client.publish(topic, payload=image_b64, qos=2)
    file.close()
    client.loop_stop()

def connect_mqtt():
    client = mqtt_client.Client(client_id="470671450867417691dd41f92aef9cb6", userdata=None, protocol=mqtt_client.MQTTv5)
    client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)
    client.username_pw_set(username, password)
    client.connect(host=host, port=port)
    return client

def publish(client, data, topic):
    result = client.publish(topic=topic, payload=str(data), qos=1)
    msg_status = result[0]
    if msg_status == 0:
        print(f"message sent to topic {topic}")
    else:
        print(f"Failed to send message to topic {topic}")

def publish_mqtt(number_of_people,topic):
    client = connect_mqtt()
    client.loop_start()
    publish(client,number_of_people,topic)
    client.loop_stop()

minute_interval = 0
hour_interval = 0
start_time_minute = time.time()
start_time_hour = time.time()
people = []
people_in_5s = 0
cam = cv2.VideoCapture(0)
ret = True
detected_person = dict()
while ret:
    ret, frame = cam.read()
    results = model.track(frame, stream=True, persist=True)
    for res in results:
        is_person = False
        now = datetime.now()
        year, month, day, hour, minute, second = (
            str(now.year),
            str(now.month),
            str(now.day),
            str(now.hour),
            str(now.minute),
            str(now.second),
        )
        for box in res.boxes:
            if int(box.cls) == 0 and float(box.conf) > threshold:
                is_person = True
                if box.id == None:
                    continue
                id_person = int(box.id[0])
                confidence = float(box.conf)
                if id_person not in people:
                    people.append(id_person)                    
                if id_person not in detected_person:
                    detected_person[id_person] = confidence
                elif detected_person.get(id_person) < confidence:
                    detected_person[id_person] = confidence
                    thread = threading.Thread(target=send_image, args=(box, frame, now))
                    thread.start()

        current_time_minute = time.time() - start_time_minute
        current_time_hour = time.time() - start_time_hour
        if current_time_minute >= 5:
            if len(people) > people_in_5s:
                people_in_5s = len(people) - people_in_5s
            minute_interval = 0
            start_time_minute = time.time()
            topic = "detected/" + day + "-" + month + "-" +  year + "/people_in_5s/" + hour + "h" + minute + "m" + second + "s"
            thread = threading.Thread(target=publish_mqtt, args=(people_in_5s,topic))
            thread.start()

        if current_time_hour >= 60:
            print(people)
            people_in_minute = len(people)
            people.clear()
            people_in_5s = 0
            hour_interval = 0
            start_time_hour = time.time()
            topic = "detected/" + day + "-" + month + "-" +  year + "/people_in_minute/" + hour + "h" + minute + "m"
            thread = threading.Thread(target=publish_mqtt, args=(people_in_minute,topic))
            thread.start()

        if is_person == True:
            res_plotted = res.plot()
        else:
            res_plotted = frame
        # time.sleep(2)
        # Cập nhật số người trong 1 phút và 1 giờ

    cv2.imshow("Webcam", res_plotted)
    if cv2.waitKey(1) == 27:  # esc
        break
cv2.destroyAllWindows()
cam.release()
detected_person.clear()   
