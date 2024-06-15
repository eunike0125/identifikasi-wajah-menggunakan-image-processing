import tkinter as tk
import cv2
import os
from PIL import Image, ImageTk
import numpy as np
import mysql.connector
from tkinter import messagebox

window = tk.Tk()
window.title("Project") 

canvas1 = tk.Canvas(window, width=490, height=490, bg='ivory')
canvas1.place(x=5, y=50)

# Entry widgets
labels = ["NAMA", "UMUR", "ALAMAT", "TTL"]
entries = []
for i, text in enumerate(labels):
    tk.Label(canvas1, text=text, font=("Arial", 13)).place(x=5, y=5 + i*50)
    entry = tk.Entry(canvas1, width=50, bd=5)
    entry.place(x=140, y=10 + i*50)
    entries.append(entry)

def train_classifier():
    data_dir = "D:/EUN/skripsi/data"
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
    messagebox.showinfo('Result', 'Training dataset completed!')

def generate_dataset():
    if any(entry.get() == "" for entry in entries):
        messagebox.showinfo('Result', 'WAJIB MENGISI SELURUH DATA!')
        return

    try:
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="",
            database="face_p"
        )
        mycursor = mydb.cursor()
        mycursor.execute("SELECT MAX(id) FROM my_table")
        max_id_result = mycursor.fetchone()

        last_id = max_id_result[0] if max_id_result[0] is not None else 0
        new_id = last_id + 1
        datagaa = 1
        namafile = f"data/user.{new_id}.{datagaa}.jpg"

        sql = "INSERT INTO my_table (id, name, age, address, ttl, datagambar) VALUES (%s, %s, %s, %s, %s, %s)"
        val = (new_id, entries[0].get(), entries[1].get(), entries[2].get(), entries[3].get(), namafile)
        mycursor.execute(sql, val)
        mydb.commit()
        mycursor.close()
        mydb.close()

        face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        def face_cropped(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                return None
            for (x, y, w, h) in faces:
                cropped_face = img[y:y+h, x:x+w]
            return cropped_face

        cap = cv2.VideoCapture(0)
        img_id = 0

        while True:
            ret, frame = cap.read()
            if face_cropped(frame) is not None:
                img_id += 1
                face = cv2.resize(face_cropped(frame), (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                file_name_path = f"data/user.{new_id}.{img_id}.jpg"
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Cropped face", face)
                if cv2.waitKey(1) == 13 or img_id == 200:
                    break

        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo('Result', 'Generating dataset completed!')

    except mysql.connector.Error as err:
        messagebox.showinfo('Result', f'MySQL Error: {str(err)}')

def detect_face():
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)
        coords = []

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            id, pred = clf.predict(gray_image[y:y+h, x:x+w])
            confidence = int(100 * (1 - pred / 300))

            mydb = mysql.connector.connect(
                host="localhost",
                user="root",
                passwd="",
                database="face_p"
            )
            # Asumsi bahwa `mydb` telah diinisialisasi sebelumnya

                # Menggunakan satu kursor untuk kedua operasi
            mycursor = mydb.cursor()

                # Menggunakan parameterisasi untuk menghindari SQL Injection
            mycursor.execute("SELECT name FROM my_table WHERE id=%s", (id,))
            s = mycursor.fetchone()

                # Mengatasi kemungkinan `None` dan memastikan `s` adalah iterable
            if s is not None and isinstance(s, tuple):
                s = ''.join(s)
            else:
                s = ''  # atau nilai default lainnya yang diinginkan

                # Melanjutkan dengan query kedua
            mycursor.execute("SELECT name, age, address, ttl, datagambar FROM my_table WHERE id=%s", (id,))
            datanama = mycursor.fetchall()

                # Pastikan `datanama` tidak kosong sebelum digunakan
            if datanama:
                    # Lanjutkan dengan kode Anda yang lain
                 pass
            else:
                print("Data tidak ditemukan")

            if confidence > 74:
                cv2.putText(img, s, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)   
                tk.Label(canvas3, text="Nama =", font=("Arial", 14)).place(x=5, y=205)        
                tk.Label(canvas3, text=datanama[0][0], font=("Arial", 14)).place(x=75, y=205)  
                tk.Label(canvas3, text="Umur =", font=("Arial", 14)).place(x=5, y=255)        
                tk.Label(canvas3, text=datanama[0][1], font=("Arial", 14)).place(x=80, y=255)  
                tk.Label(canvas3, text="Alamat =", font=("Arial", 14)).place(x=5, y=305)        
                tk.Label(canvas3, text=datanama[0][2], font=("Arial", 14)).place(x=80, y=305)  
                tk.Label(canvas3, text="TTL =", font=("Arial", 14)).place(x=5, y=355)        
                tk.Label(canvas3, text=datanama[0][3], font=("Arial", 14)).place(x=80, y=355)                                      
                load4 = Image.open(datanama[0][4])
                photo5 = ImageTk.PhotoImage(load4)
                gambarrrr = tk.Label(canvas3, image=photo5, width=190, height=200)
                gambarrrr.image = photo5
                gambarrrr.place(x=0, y=5)
            else:
                cv2.putText(img, "UNKNOWN", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), "Face", clf)
        return img

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    video_capture = cv2.VideoCapture(0)
    while True:
        ret, img = video_capture.read()
        img = recognize(img, clf, faceCascade)
        cv2.imshow("Face Detection", img)
        if cv2.waitKey(1) == 13:
            break

    video_capture.release()
    cv2.destroyAllWindows()

canvas2 = tk.Canvas(window, width=500, height=350)
canvas2.place(x=5, y=300)

def capture_image():
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False

    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("tidak ada yang terdeteksi..")
        return None
    else:
        for (x, y, w, h) in faces:
            cropped_face = frame[y:y+h, x:x+w]
        cv2.imwrite("captured_image.jpg", cropped_face)

        load = Image.open("captured_image.jpg")
        photo = ImageTk.PhotoImage(load)

        img = tk.Label(canvas3, image=photo, width=200, height=200)
        img.image = photo
        img.place(x=0, y=5)

        cap.release()

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    for (x, y, w, h) in faces:     
        id, pred = clf.predict(gray[y:y+h, x:x+w])

        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="",
            database="face_p"
        )
        mycursor = mydb.cursor()
        mycursor.execute("SELECT name, age, address FROM my_table WHERE id=" + str(id))
        s = mycursor.fetchall()

        tk.Label(canvas3, text="Nama =", font=("Arial", 20)).place(x=5, y=250)
        tk.Label(canvas3, text=s[0][0], font=("Arial", 20)).place(x=100, y=250)    
        tk.Label(canvas3, text="Umur =", font=("Arial", 20)).place(x=5, y=300)  
        tk.Label(canvas3, text=s[0][1], font=("Arial", 20)).place(x=100, y=300)             
        tk.Label(canvas3, text="Alamat =", font=("Arial", 20)).place(x=5, y=350)                        
        tk.Label(canvas3, text=s[0][2], font=("Arial", 20)).place(x=150, y=350)

b1 = tk.Button(canvas2, text="Simpan Data Gambar", font=("Arial", 16), bg='grey', fg='white', command=train_classifier)
b1.place(x=5, y=65)

b2 = tk.Button(canvas2, text="Simpan Ke Database", font=("Arial", 16), bg='pink', fg='black', command=generate_dataset)
b2.place(x=5, y=15)

b4 = tk.Button(canvas2, text="Prediksi", font=("Arial", 20), bg="cyan", fg="black", command=detect_face)
b4.place(x=5, y=125)

load3 = Image.open('3183.jpg')
photo3 = ImageTk.PhotoImage(load3)

canvas3 = tk.Canvas(window, width=700, height=600)
canvas3.place(x=515, y=50)
canvas3.create_image(140, 265, image=photo3)

window.geometry("800x680")
window.mainloop()
