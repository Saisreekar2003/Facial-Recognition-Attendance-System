
import cv2
import time
import face_recognition
import numpy as np
import csv
import os
from datetime import datetime
import argparse
import warnings
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
import urllib.request as ur

from Adafruit_IO import Client, Feed

aio = Client('saisreekarnitap', 'aio_jnKE27vtdIlMdACNkDtHH7gG7wm6')

count=0

SAMPLE_IMAGE_PATH = "./images/sample/"

video_capture = cv2.VideoCapture(0)

sai_sreekar_image = face_recognition.load_image_file("images/sample/Sai Sreekar.jpeg")
sai_sreekar_encoding = face_recognition.face_encodings(sai_sreekar_image)[0]

harika_image = face_recognition.load_image_file("images/sample/har.jpeg")
harika_encoding = face_recognition.face_encodings(harika_image)[0]


sathvika_image = face_recognition.load_image_file("images/sample/sat.jpeg")
sathvika_encoding = face_recognition.face_encodings(sathvika_image)[0]

siri_image = face_recognition.load_image_file("images/sample/siri.jpg")
siri_encoding = face_recognition.face_encodings(siri_image)[0]

soma_image = face_recognition.load_image_file("images/sample/Soma.jpg")
soma_encoding = face_recognition.face_encodings(soma_image)[0]
known_face_encoding = [sai_sreekar_encoding,harika_encoding,sathvika_encoding,siri_encoding,soma_encoding]

known_faces_names = ["620217","620112","620113","620228","620255"]
s=True
students = known_faces_names.copy()
en_students=[]
check=[]
uncheck=known_faces_names.copy()
face_locations = []
face_encodings = []
face_names = []
abface_names=[]
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
f = open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)

def ada(roll,msg):
    eman=int(roll)
    if eman==620217:
        s= aio.feeds('sai')
        aio.send_data(s.key, eman)
        sai= aio.feeds('sai-sreekar')
        aio.send_data(sai.key, msg)
    elif eman==620112:
        har= aio.feeds('harika')
        aio.send_data(har.key, msg)
    elif eman==620113:
        sat= aio.feeds('sathvika')
        aio.send_data(sat.key, msg)
    elif eman==620228:
        siri= aio.feeds('siri')
        aio.send_data(siri.key, msg)
    elif eman==620255:
        soma= aio.feeds('soma')
        aio.send_data(soma.key, msg)

def rolls(roll,msg):
    msg= msg.replace(' ', "%20")
    msg= msg.replace('\n', "%0A")
    eman=int(roll)
    if eman==620217:  
        b=ur.urlopen('https://api.thingspeak.com/update?api_key=DTIIZANTOV52ST6S&field1='+msg)
    elif eman==620112:
        b=ur.urlopen('https://api.thingspeak.com/update?api_key=DTIIZANTOV52ST6S&field2='+msg)
    elif eman==620113:
        b=ur.urlopen('https://api.thingspeak.com/update?api_key=DTIIZANTOV52ST6S&field3='+msg)
    elif eman==620228:
        b=ur.urlopen('https://api.thingspeak.com/update?api_key=DTIIZANTOV52ST6S&field4='+msg)
    elif eman==620255:
        b=ur.urlopen('https://api.thingspeak.com/update?api_key=DTIIZANTOV52ST6S&field5='+msg)
    print("cloud data updated")
    
def recog():
    while True:
        _,frame = video_capture.read()
        small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
        rgb_small_frame = small_frame[:,:,::-1]
        if s:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
                name=""
                face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
                best_match_index = np.argmin(face_distance)
                if matches[best_match_index]:
                    name = known_faces_names[best_match_index]

        cv2.imshow("Attendence System",frame)
        if cv2.waitKey(1) & 0xFF == ord('r'):
            break

    return name

def cap(name):   
    cap = cv2.VideoCapture(0)
   
  
    while True:
        ret, img = cap.read()
        cv2.imshow(name, img)
  
        k = cv2.waitKey(125)

        if k == ord('c'):
  
            ret, img = cap.read()

            cv2.imwrite(name+'.jpg', img)

            rot=cv2.imread(name+'.jpg')

            image=cv2.rotate(rot,cv2.ROTATE_90_CLOCKWISE)

            cv2.imshow(name, image)

            cv2.waitKey(1000)

            cv2.imwrite(SAMPLE_IMAGE_PATH+ name+'.jpg', image)
  
        elif k == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows()


def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def test(image_name, model_dir, device_id):
    global value,label
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = cv2.imread(SAMPLE_IMAGE_PATH + image_name)
    result = check_image(image)
    if result is False:
        return
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time()-start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    if label == 1:
        print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
        result_text = "RealFace Score: {:.2f}".format(value)
        color = (255, 0, 0)
    else:
        print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
        result_text = "FakeFace Score: {:.2f}".format(value)
        color = (0, 0, 255)
    print("Prediction cost {:.2f} s".format(test_speed))
    cv2.rectangle(
        image,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    cv2.putText(
        image,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)

    format_ = os.path.splitext(image_name)[-1]
    result_image_name = image_name.replace(format_, "_result" + format_)
    cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image)

def pre():
    global count
    count=count+1
    while True:
        _,frame = video_capture.read()
        small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
        rgb_small_frame = small_frame[:,:,::-1]
        if s:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
                face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
                best_match_index = np.argmin(face_distance)
                if matches[best_match_index]:
                    name = known_faces_names[best_match_index]
 
                face_names.append(name)
                if name in known_faces_names:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (10,100)
                    fontScale              = 1
                    fontColor              = (0,255,0)
                    thickness              = 3
                    lineType               = 2
 
                    cv2.putText(frame,name+' Entry',bottomLeftCornerOfText, font, fontScale,fontColor,thickness,lineType)
 
                    if name in students:
                        students.remove(name)
                        if name not in en_students:
                            en_students.append(name)                        
                        print("Students Left For Entry")
                        print(students)
                        print("Students in Class and Left For Exit")
                        if count!=0:
                            print(en_students)
                        else:
                            print("No one in the class")
                        enow=datetime.now()
                        en_time = enow.strftime("%H-%M-%S")
                        ent="Entered"
                        msg=name+"  "+ent
                        rolls(name,msg)
                        ada(name,"Joined The Class")
                        lnwriter.writerow([name,"Entered at",en_time])
  
        cv2.imshow("Attendence System",frame)
        if cv2.waitKey(1) & 0xFF == ord('e'):
                break
    uncheck.remove(name)
    check.append(name)

def abse():
    global count
    count=count-1
    while True:
        _,frame = video_capture.read()
        small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
        rgb_small_frame = small_frame[:,:,::-1]
        if s:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
            abface_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
                face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
                best_match_index = np.argmin(face_distance)
                if matches[best_match_index]:
                    name = known_faces_names[best_match_index]
 
                abface_names.append(name)
                if name in known_faces_names:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (10,100)
                    fontScale              = 1
                    fontColor              = (0,0,255)
                    thickness              = 3
                    lineType               = 2
 
                    cv2.putText(frame,name+' Exit',bottomLeftCornerOfText, font, fontScale,fontColor,thickness,lineType)

                    if name in en_students:
                        en_students.remove(name)
                        if name not in students:
                            students.append(name)
                        print("Students Left For Entry")
                        print(students)
                        print("Students Left For Exit")
                        if count!=0:
                            print(en_students)
                        else:
                            print("No one in the class")
                        exnow=datetime.now()
                        ex_time = exnow.strftime("%H-%M-%S")
                        et="Exited"
                        msg=name+"   "+et
                        rolls(name,msg)
                        ada(name,"Left The Class")
                        lnwriter.writerow([name,"Exited at",ex_time])

        cv2.imshow("Attendence System",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    check.remove(name)
    uncheck.append(name)

while True :
    global value,label
    nammmm=recog()
    cap(recog())
    if __name__ == "__main__":
        desc = "test"
        parser = argparse.ArgumentParser(description=desc)
        parser.add_argument(
            "--device_id",
            type=int,
            default=0,
            help="which gpu id, [0/1/2/3]")
        parser.add_argument(
            "--model_dir",
            type=str,
            default="./resources/anti_spoof_models",
            help="model_lib used to test")
        parser.add_argument(
            "--image_name",
            type=str,
            default=nammmm+".jpg",
            help="image used to test")
        args = parser.parse_args()
        test(args.image_name, args.model_dir, args.device_id)
        video_capture = cv2.VideoCapture(0)
        if label==1:
            if value>=0.5:
                while True:
                    if  cv2.waitKey(5000) & 0xFF == ord('x'):
                        break
                    else:
                        if recog() in uncheck:
                            pre()
                        else:
                            abse()
            else:
                print("FAKE FACE")
        else:
            print("FAKE FACE")
        
video_capture.release()
cv2.destroyAllWindows()
f.close()

