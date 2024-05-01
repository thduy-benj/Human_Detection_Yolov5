import os
import cv2
import time
import threading
from ultralytics import YOLO
from datetime import datetime
from paho.mqtt import client as mqtt_client

model=YOLO("yolov5nu.pt")
broker='broker.hivemq.com'
port=1883
threshold=0.7 # Ngưỡng chấp nhận để gửi ảnh (%)

# Hàm gửi ảnh (lưu vào local và gửi qua mqtt)
def send_image(box,frame):
    x1, y1, x2, y2= box.xyxy[0] # lưu thông số kích thước của box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    crop_person = frame[y1:y2,x1:x2] # cắt phần ảnh có người
    id_person = str(int(box.id[0]))
    now = datetime.now()
    year,month,day,hour,minute,second = str(now.year),str(now.month),str(now.day),str(now.hour),str(now.minute),str(now.second)
    file_path = ".\\detected" # Tạo thư mục detect
    if os.path.exists(file_path) == False:
        os.mkdir(file_path) 
    file_path += "\\" + day + "-" + month + "-" + year # mỗi ngày là một thư mực
    if os.path.exists(file_path) == False:
        os.mkdir(file_path)
    file_path += "\\" + "ID" + id_person # thư mục mỗi ngày chứa các thư mục ID người
    if os.path.exists(file_path) == False:
        os.mkdir(file_path)

    # cái này để lưu ảnh vào local 
    image_name = file_path + "\\" + hour + "-" + minute + "-" + second + "conf " + str(float(box.conf)) + ".jpg"
    is_saved = cv2.imwrite(image_name,crop_person)
    if is_saved: print("ảnh đã được lưu") 
    else: print("ERROR")

    # này gửi ảnh qua mqtt
    client = connect_mqtt()
    client.loop_start()
    publish(client=client,byteArr=crop_person.tobytes(),topic=file_path)
    time.sleep(1)
    client.loop_stop()

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Successfully connected to MQTT broker")
        else:
            print("Failed to connect, return code %d", rc)
    client = mqtt_client.Client()
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def publish(client,byteArr,topic):
    result = client.publish(topic,byteArr,2)
    msg_status = result[0]
    if msg_status == 0:
        print(f"message sent to topic {topic}")
    else:
        print(f"Failed to send message to topic {topic}")

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    ret = True
    detected_person = dict()
    while ret:
        ret, frame = cam.read()
        results = model.track(frame, stream=True, persist=True)
        for res in results:
            is_person = False
            for box in res.boxes:
                if (int(box.cls) == 0 and float(box.conf) > threshold):
                    is_person = True
                    if (box.id[0]==None): continue # cái này để đề phòng lỗi th chứ ko có ý nghĩa gì
                    id_person = int(box.id[0]) 
                    confidence = float(box.conf)
                    if id_person not in detected_person:
                        detected_person[id_person] = confidence
                    # nếu conf cũ nhỏ hơn thì cập nhật gửi frame mới và cập nhật conf hiện tại
                    elif detected_person.get(id_person) < confidence:
                        detected_person[id_person] = confidence
                        thread = threading.Thread(target=send_image, args=(box,frame))
                        thread.start()
            # nếu có người thì lấy ảnh có box còn ko có người thì lấy frame ban đầu
            # nếu muốn chạy nhanh hơn thì comment lại vì không phải show ảnh lên làm gì 
            if is_person == True:
                res_plotted = res.plot()
            else:
                res_plotted = frame
        cv2.imshow('Webcam', res_plotted)
        if cv2.waitKey(1) == 27: #esc
            break
    cv2.destroyAllWindows()
    cam.release()
    detected_person.clear()