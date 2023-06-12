import sys
import tkinter as tk
from tkinter import ttk
import cv2 as cv
from PIL import Image, ImageTk
import os
import numpy as np


class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.output_label = None
        self.geometry("900x720")
        self.title('THÀNH ĐỊT')
        self.languages = ('Option 1', 'Option 2', 'Option 3', 'Option 4')
        self.option_var = tk.StringVar(self)
        self.create_wigets()

    def ngu_lol(self):
        cam = cv.VideoCapture(0)
        filename = "haarcascade_frontalface_default.xml"
        if not cam.isOpened():
            print("Cam như lồn")
            exit()
        while True:
            ret, frame = cam.read()
            if not ret:
                print("đéo đọc được khung hình nào!")
                break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            face_cascade = cv.CascadeClassifier(filename)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)
            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.imshow('frame', frame)
            if cv.waitKey(1) == ord('q'):
                break
        cam.release()
        cv.destroyAllWindows()

    def create_wigets(self):
        file = Image.open('bg1.jpg')
        bg = ImageTk.PhotoImage(file)
        label = tk.Label(image=bg)
        label.file = bg
        label.grid(row=0, column=0, columnspan=500, rowspan=500, padx=1, pady=1)
        paddings = {'padx': 10, 'pady': 10}
        label = ttk.Label(self, text='Select your option:')
        label.grid(column=0, row=0, sticky=tk.W, **paddings)
        option_menu = ttk.OptionMenu(self, self.option_var, self.languages[0], *self.languages,
                                     command=self.option_changed)
        option_menu.grid(column=1, row=0, sticky=tk.W, **paddings)
        self.output_label = ttk.Label(self, foreground='red')
        self.output_label.grid(column=0, row=2, sticky=tk.W, **paddings)

    def option_changed(self, *args):
        if self.option_var.get() == 'Option 1':
            self.after(0, self.ngu_lol(), *args)
        if self.option_var.get() == 'Option 2':
            self.after(0, self.info_ob(), *args)
        if self.option_var.get() == 'Option 3':
            self.after(0, self.reg(), *args)
        if self.option_var.get() == 'Option 4':
            self.after(0, self.close(), *args)

    def close(self):
        self.destroy()
        self.quit()
        sys.exit()

    def info_ob(self):
        master = self

        l1 = ttk.Label(master, text="Enter your name:")
        l2 = ttk.Label(master, text="Enter your ID:")

        l1.grid(row=1, column=0, sticky=tk.W, pady=2)
        l2.grid(row=2, column=0, sticky=tk.W, pady=2)

        e1 = ttk.Entry(master)
        e2 = ttk.Entry(master)

        e1.grid(row=1, column=1, pady=2)
        e2.grid(row=2, column=1, pady=2)

        def crea():
            folder_name = e2.get() + '.' + e1.get()
            path = os.path.join("C:/Users/Trang/PycharmProjects/pythonProject/asm/dataSet/", folder_name)
            try:
                os.mkdir(path)
            except FileExistsError:
                print("folder already exists.")
            b3.grid_remove()
            return path

        b1 = ttk.Button(master, text="ok", command=lambda: fina())
        b2 = ttk.Button(master, text="exit", command=lambda: remove())
        b3 = ttk.Button(master, text="create", command=lambda: crea())
        b4 = ttk.Button(master, text="training", command=lambda: train())

        b1.grid(row=4, column=0, sticky=tk.E)
        b2.grid(row=4, column=1, sticky=tk.E)
        b3.grid(row=4, column=2, sticky=tk.E)
        b4.grid(row=5, column=0, sticky=tk.E)

        def fina():

            cap = cv.VideoCapture(0)
            filename = "haarcascade_frontalface_default.xml"
            if not cap.isOpened():
                print("Cam như lồn")
                exit()
            count = 0
            cout = 0
            label = None
            while True:
                cout += 1
                self.update()
                if cout > 1:
                    label.grid_remove()
                ret, img = cap.read()
                img = cv.flip(img, 1)
                face_cascade = cv.CascadeClassifier(filename)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    count += 1
                    cv.imwrite(os.path.join(crea(), 'face' + '.' + e2.get() + '.' + str(count) + ".jpg"),
                               gray[y:y + h,x : x+ w])
                if count >= 80:
                    h = ttk.Button(master, text="done", command=lambda: qut())
                    h.grid(row=5, column=2, sticky=tk.E)
                    break
                    quit()
                cv2image = cv.cvtColor(img, cv.COLOR_BGR2RGBA)
                img1 = Image.fromarray(cv2image)
                img = ImageTk.PhotoImage(image=img1)
                label = tk.Label(image=img)
                label.img = img
                label.configure(image=img)
                label.grid(row=0, column=6, columnspan=50, rowspan=50, padx=2, pady=2)

            def qut():
                if h:
                    h.grid_remove()

            return

        def remove():
            l1.grid_remove()
            l2.grid_remove()
            e1.grid_remove()
            e2.grid_remove()
            b1.grid_remove()
            b2.grid_remove()
            b4.grid_remove()

        def train():
            width_d, height_d = 280, 280
            fold_name = e1.get() + '.' + e2.get()
            path = 'C:/Users/Trang/PycharmProjects/pythonProject/asm/dataSet/'
            recognizer = cv.face.LBPHFaceRecognizer_create()
            detector = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

            def getImagesAndLabels(path):
                infoPaths = [os.path.join(path, f) for f in os.listdir(path)]
                faceSamples = []
                ids = []
                for infoPath in infoPaths:
                    imagePaths = [os.path.join(infoPath, f) for f in os.listdir(infoPath)]
                    for imagePath in imagePaths:

                        PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
                        img_numpy = np.array(PIL_img, 'uint8')

                        id = int(os.path.split(imagePath)[-1].split(".")[1])
                        faces = detector.detectMultiScale(img_numpy)

                        for (x, y, w, h) in faces:
                            faceSamples.append(cv.resize(img_numpy[y:y + h, x:x + w], (width_d, height_d)))
                            ids.append(id)

                return faceSamples, ids

            faces, ids = getImagesAndLabels(path)
            recognizer.train(faces, np.array(ids))

            # Save the model into trainer/trainer.yml
            names = e1.get() + '.' + e2.get()
            recognizer.save(
                'C:/Users/Trang/PycharmProjects/pythonProject/asm/trainer/' +'face'+ '.yml')
            b5 = ttk.Button(master, text="done",command=lambda: clea() )
            b5.grid(row=5, column=0, sticky=tk.E)

            def clea():
                b5.grid_remove()

    def reg(self):
        import cv2
        import numpy as np
        import os
        pat = 'asm/dataSet/'
        list_train = [os.path.join(pat, f) for f in os.listdir(pat)]
        recognizer1 = cv2.face.LBPHFaceRecognizer_create()
        cascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath)
        names = ['None']
        couq = 0
        for i in list_train:
            a = str(list_train[couq].split('.', -1)[1]).replace('asm/dataSet/', '')
            print(a)
            names.append(a)
            couq += 1
        recognizer1.read('C:/Users/Trang/PycharmProjects/pythonProject/asm/trainer/face.yml')

        font = cv2.FONT_HERSHEY_SIMPLEX

        # iniciate id counter
        id = 0

        # Initialize and start realtime video capture
        cam = cv2.VideoCapture(0)

        # Define min window size to be recognized as a face
        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)

        while True:

            ret, img = cam.read()
            # img = cv2.flip(img, -1) # Flip vertically

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )

            for (x, y, w, h) in faces:

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                id, confidence = recognizer1.predict(gray[y:y + h, x:x + w])

                # Check if confidence is less them 100 ==> "0" is perfect match

                if 100 > confidence > 45:
                    id = names[id]
                else:
                    id = "unknown"
                cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)

            cv2.imshow('camera', img)

            k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
            if cv.waitKey(1) == ord('q'):
                break

        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = App()
    app.mainloop()
