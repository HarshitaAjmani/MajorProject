# Importing Libraries

import numpy as np

import cv2
import os, sys
import time
import operator

from string import ascii_uppercase

import tkinter as tk
from PIL import Image, ImageTk

from hunspell import Hunspell
import enchant

from keras.models import model_from_json

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

#Application :

class Application:

    def __init__(self):

        self.hs = Hunspell('en_US')  #This line initializes an object named hs of the Hunspell class. It seems to be a spell-checking library. It's initialized with the English language dictionary (en_US).
        self.vs = cv2.VideoCapture(0) #Here, an object named vs is initialized from the VideoCapture class of the OpenCV library (cv2). It's used to capture video frames from a camera. The argument 0 indicates the default camera.
        self.current_image = None
        self.current_image2 = None #these two variale likely be used to store current video frames or images.


        #Here, a JSON file containing a model architecture definition is opened and read. The file path is specified. The content of the file is read into the variable model_json, and then the file is closed.
        self.json_file = open("D:\\SRM\\Assingment\\8Sem\\Review\\2nd_git\\two\\Models\\model_new.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()

        self.loaded_model = model_from_json(self.model_json)
        self.loaded_model.load_weights("Models\model_new.h5")

        self.json_file_dru = open("Models\model-bw_dru.json" , "r")
        self.model_json_dru = self.json_file_dru.read()
        self.json_file_dru.close()

        self.loaded_model_dru = model_from_json(self.model_json_dru)
        self.loaded_model_dru.load_weights("Models\model-bw_dru.h5")
        self.json_file_tkdi = open("Models\model-bw_tkdi.json" , "r")
        self.model_json_tkdi = self.json_file_tkdi.read()
        self.json_file_tkdi.close()

        self.loaded_model_tkdi = model_from_json(self.model_json_tkdi)
        self.loaded_model_tkdi.load_weights("Models\model-bw_tkdi.h5")
        self.json_file_smn = open("Models\model-bw_smn.json" , "r")
        self.model_json_smn = self.json_file_smn.read()
        self.json_file_smn.close()

        self.loaded_model_smn = model_from_json(self.model_json_smn)
        self.loaded_model_smn.load_weights("Models\model-bw_smn.h5")

        
        #Here, a dictionary ct is initialized with a key 'blank' set to 0. 
        #Another variable blank_flag is also initialized to 0. These variables will likely be used to keep counts of different symbols recognized.
        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0

        for i in ascii_uppercase:
          self.ct[i] = 0
        
        print("Loaded model from disk")

        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("900x900")

        self.panel = tk.Label(self.root)
        self.panel.place(x = 100, y = 10, width = 580, height = 580)
        
        self.panel2 = tk.Label(self.root) # initialize image panel
        self.panel2.place(x = 400, y = 65, width = 275, height = 275)

        self.T = tk.Label(self.root)
        self.T.place(x = 60, y = 5)
        self.T.config(text = "Sign Language To Text Conversion", font = ("Courier", 30, "bold"))

        self.panel3 = tk.Label(self.root) # Current Symbol
        self.panel3.place(x = 500, y = 540)

        self.T1 = tk.Label(self.root)
        self.T1.place(x = 10, y = 540)
        self.T1.config(text = "Character :", font = ("Courier", 30, "bold"))

        self.panel4 = tk.Label(self.root) # Word
        self.panel4.place(x = 220, y = 595)

        self.T2 = tk.Label(self.root)
        self.T2.place(x = 10,y = 595)
        self.T2.config(text = "Word :", font = ("Courier", 30, "bold"))

        self.panel5 = tk.Label(self.root) # Sentence
        self.panel5.place(x = 350, y = 645)

        self.T3 = tk.Label(self.root)
        self.T3.place(x = 10, y = 645)
        self.T3.config(text = "Sentence :",font = ("Courier", 30, "bold"))

        self.T4 = tk.Label(self.root)
        self.T4.place(x = 250, y = 690)
        self.T4.config(text = "Suggestions :", fg = "red", font = ("Courier", 30, "bold"))

        self.bt1 = tk.Button(self.root, command = self.action1, height = 0, width = 0)
        self.bt1.place(x = 26, y = 745)

        self.bt2 = tk.Button(self.root, command = self.action2, height = 0, width = 0)
        self.bt2.place(x = 325, y = 745)

        self.bt3 = tk.Button(self.root, command = self.action3, height = 0, width = 0)
        self.bt3.place(x = 625, y = 745)

        #The variables like self.str, self.word, self.current_symbol, self.photo are initialized to hold certain values and states related to the application's functionality. These variables will be updated and used as the application runs.
        self.str = ""
        self.word = " "
        self.current_symbol = "Empty"
        self.photo = "Empty"
        self.video_loop()
        #the method self.video_loop() is called to start a loop that continuously captures video frames and processes them for sign language recognition


    def video_loop(self):
        ok, frame = self.vs.read()
 #This line reads a frame from the video capture (self.vs). 
#It returns two values: ok, a boolean indicating whether the frame was successfully read, and frame, the actual frame itself.

        if ok:
            cv2image = cv2.flip(frame, 1)
            # it flips the frame horizontally using OpenCV's cv2.flip function. This is commonly done to correct the orientation of the video stream.

            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])
            # define the coordinates for a rectangle that will be drawn on the frame. This rectangle is likely used as a bounding box for a region of interest (ROI).

            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0) ,1)  #This line draws a rectangle on the frame using OpenCV's cv2.rectangle function. The rectangle is drawn with blue color (255, 0, 0) and thickness 1.
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA) #This converts the color space of the frame from BGR (Blue, Green, Red) to RGBA (Red, Green, Blue, Alpha).

            self.current_image = Image.fromarray(cv2image) #convert the OpenCV image to a PIL (Python Imaging Library) image 
            imgtk = ImageTk.PhotoImage(image = self.current_image)

            self.panel.imgtk = imgtk
            self.panel.config(image = imgtk)


            # image augmentation
            cv2image = cv2image[y1 : y2, x1 : x2]

            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 3 )

            th3 = cv2.adaptiveThreshold(blur, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2) #adaptive thresholding to extract hands from the background.
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            self.predict(res) #It predicts the sign language gesture using the trained model

            self.current_image2 = Image.fromarray(res)

            imgtk = ImageTk.PhotoImage(image = self.current_image2)

            self.panel2.imgtk = imgtk
            self.panel2.config(image = imgtk)

            self.panel3.config(text = self.current_symbol, font = ("Courier", 30))

            self.panel4.config(text = self.word, font = ("Courier", 30))

            self.panel5.config(text = self.str,font = ("Courier", 30))

            predicts = self.hs.suggest(self.word) #suggestions obtained from the spell-checker

            # character > 60 -> word -> suggestion

            if(len(predicts) > 1):

                self.bt1.config(text = predicts[0], font = ("Courier", 20))

            else:

                self.bt1.config(text = "")

            if(len(predicts) > 2):

                self.bt2.config(text = predicts[1], font = ("Courier", 20))

            else:

                self.bt2.config(text = "")

            if(len(predicts) > 3):

                self.bt3.config(text = predicts[2], font = ("Courier", 20))

            else:

                self.bt3.config(text = "")


        self.root.after(5, self.video_loop)
        #This line schedules the video_loop method to be called again after 5 milliseconds using Tkinter's after method. This creates a loop that continuously captures and processes video frames.

    def predict(self, test_image):

        test_image = cv2.resize(test_image, (128, 128))
        #This line resizes the input test image to a size of 128x128 pixels. This is likely done to match the input size expected by the neural network models.
        
        # new model is the trained one
        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))
        #Here, the resized test image is reshaped to fit the input dimensions expected by the loaded Keras model (loaded_model). Then, the model's predict method is used to obtain the prediction for the input image.

        result_dru = self.loaded_model_dru.predict(test_image.reshape(1 , 128 , 128 , 1))

        result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1 , 128 , 128 , 1))

        result_smn = self.loaded_model_smn.predict(test_image.reshape(1 , 128 , 128 , 1))

        #Here, a dictionary prediction is initialized to store the predictions for each class. The predictions from the first model (loaded_model) are stored in this dictionary.
        prediction = {}

        prediction['blank'] = result[0][0]

        inde = 1
        
        for i in ascii_uppercase:
            
            prediction[i] = result[0][inde]

            inde += 1

        # through this for loop predicted value of every alphabet is stored in the dict prediction
        # example {a: 4 , p: 6,   t: 3}
            
        #LAYER 1

        prediction = sorted(prediction.items(), key = operator.itemgetter(1), reverse = True)

        self.current_symbol = prediction[0][0]
        #The predictions are sorted in descending order based on their probabilities, and the most probable symbol is assigned to self.current_symbol.

        #LAYER 2

        if(self.current_symbol == 'D' or self.current_symbol == 'R' or self.current_symbol == 'U'):

            prediction = {}

            prediction['D'] = result_dru[0][0]
            prediction['R'] = result_dru[0][1]
            prediction['U'] = result_dru[0][2]

            prediction = sorted(prediction.items(), key = operator.itemgetter(1), reverse = True)
            
            self.current_symbol = prediction[0][0]

        if(self.current_symbol == 'D' or self.current_symbol == 'I' or self.current_symbol == 'K' or self.current_symbol == 'T'):

            prediction = {}

            prediction['D'] = result_tkdi[0][0]
            prediction['I'] = result_tkdi[0][1]
            prediction['K'] = result_tkdi[0][2]
            prediction['T'] = result_tkdi[0][3]
            
            prediction = sorted(prediction.items(), key = operator.itemgetter(1), reverse = True)

            self.current_symbol = prediction[0][0]

        if(self.current_symbol == 'M' or self.current_symbol == 'N' or self.current_symbol == 'S'):

            prediction1 = {}

            prediction1['M'] = result_smn[0][0]
            prediction1['N'] = result_smn[0][1]
            prediction1['S'] = result_smn[0][2]

            prediction1 = sorted(prediction1.items(), key = operator.itemgetter(1), reverse = True)

            if(prediction1[0][0] == 'S'):
                
                self.current_symbol = prediction1[0][0]
    
            else:

                self.current_symbol = prediction[0][0]
        
        #This conddition is used to refresh the count of characters (Basically to detect new change in character)
        if(self.current_symbol == 'blank'):

            for i in ascii_uppercase:
                self.ct[i] = 0

        self.ct[self.current_symbol] += 1
        #This line increments the counter for the current symbol (self.current_symbol) in the self.ct dictionary.

        print(self.ct[self.current_symbol], self.current_symbol)

        #This line checks if the counter for the current symbol exceeds a threshold value (in this case, 10). If it does, it performs further checks and actions based on the counter values.
        if(self.ct[self.current_symbol] > 10):
            
            #This for loop is to remove confusion 
            #to increase or decrease the confusion
            for i in ascii_uppercase:
                #Skip the iteration if the variable i is current symbol, but continue with the next iteration:
                if i == self.current_symbol:        
                    continue                        

                tmp = self.ct[self.current_symbol] - self.ct[i]

                if tmp < 0:
                    tmp *= -1

                if tmp <= 1:
                    self.ct['blank'] = 0

                    for i in ascii_uppercase:
                        self.ct[i] = 0
                    return

            
            #Blank is used to add the word in sentence
            if self.current_symbol == 'blank':

                if self.blank_flag == 0:
                    self.blank_flag = 1

                    if len(self.str) > 0:
                        self.str += " "

                    self.str += self.word

                    self.word = ""
                    

            else:

                if(len(self.str) > 16):
                    self.str = ""

                self.blank_flag = 0

                self.word += self.current_symbol

            self.ct['blank'] = 0

            for i in ascii_uppercase:
               self.ct[i] = 0

    def action1(self):

        predicts = self.hs.suggest(self.word)
        
        if(len(predicts) > 0):

            self.word = ""

            self.str += " "

            self.str += predicts[0]

    def action2(self):

        predicts = self.hs.suggest(self.word)

        if(len(predicts) > 1):
            self.word = ""
            self.str += " "
            self.str += predicts[1]

    def action3(self):

        predicts = self.hs.suggest(self.word)

        if(len(predicts) > 2):
            self.word = ""
            self.str += " "
            self.str += predicts[2]

    def action4(self):

        predicts = self.hs.suggest(self.word)

        if(len(predicts) > 3):
            self.word = ""
            self.str += " "
            self.str += predicts[3]

    def action5(self):

        predicts = self.hs.suggest(self.word)

        if(len(predicts) > 4):
            self.word = ""
            self.str += " "
            self.str += predicts[4]
            
    def destructor(self):

        print("Closing Application...")

        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()
    
print("Starting Application...")

(Application()).root.mainloop()