import os
import cv2
from datetime import datetime
import paho.mqtt.client as mqtt_client
from paho import mqtt
import numpy as np
import base64

host = '470671450867417691dd41f92aef9cb6.s1.eu.hivemq.cloud'
port = 8883
username = "hivemq.webclient.1714738779741"
password = "!zm>.y14Z2BbuD5f<YNL"

now = datetime.now()
year, month, day = str(now.year), str(now.month), str(now.day)
topic_today = day + "-" + month + "-" + year

def on_message(client, userdata, msg):
    topic = msg.topic
    payload = msg.payload
    print(topic)
    image_data = base64.b64decode(payload)
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Hiển thị ảnh trong một cửa sổ mới
    # cv2.imshow("Received Image", img)
    # cv2.destroyAllWindows()
    global year,month,day,hour,minute,second,now
    now = datetime.now()
    year, month, day, hour, minute, second = (
        str(now.year),
        str(now.month),
        str(now.day),
        str(now.hour),
        str(now.minute),
        str(now.second),
    )
    file_path = ".\\receive"  # Tạo thư mục detect
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    file_path += "\\" + day + "-" + month + "-" + year  # mỗi ngày là một thư mực
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    image_name = (
        file_path
        + "\\"
        + hour
        + "h"
        + minute
        + "m"
        + second
        + "s "
        + ".jpg"
    )
    is_saved = cv2.imwrite(image_name, img)
    if is_saved:
        print("ảnh đã được lưu")
    else:
        print("ERROR")
def subscribe(client):
    global topic_today
    topic_today = day + "-" + month + "-" + year
    client.subscribe("detected/"+topic_today+"/image/#")
    client.on_message = on_message
def connect_mqtt():
    client = mqtt_client.Client(client_id="470671450867417691dd41f92aef9cb6", userdata=None, protocol=mqtt_client.MQTTv5)
    client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)
    client.username_pw_set(username, password)
    client.connect(host=host, port=port)
    return client

def run():
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()
if __name__ == '__main__':
    run()