# Import required modules
from tensorflow import keras
import numpy as np
from PyQt5 import * #gui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys # python native library
import subprocess # run multiple code such as camera
import pyttsx3 # sound audio
from threading import Thread # run proccess without blocking

# Initiate Text-to-Speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# Load both the models
model = keras.models.load_model('traffic_classifier.h5')
detector_model = keras.models.load_model(f"traffic_detector.h5")

# We create the list to store sequence of class names. 
# The model only gives index number for each class, we we need to know class names based on that index
classes = {
    0:['No traffic sign', 'Traffic sign not detected, Please upload another image'],
    1:['Speed limit (20km/h)', 'You should stay below 20 killometers per hour'],
    2:['Speed limit (30km/h)', 'You should stay below 30 killometers per hour'],
    3:['Speed limit (50km/h)', 'You should stay below 50 killometers per hour'],
    4:['Speed limit (60km/h)', 'You should stay below 60 killometers per hour'],
    5:['Speed limit (70km/h)', 'You should stay below 70 killometers per hour'],
    6:['Speed limit (80km/h)', 'You should stay below 80 killometers per hour'],
    7:['End of speed limit (80km/h)', 'You should stay below 80 killometers per hour'],
    8:['Speed limit (100km/h)', 'You should stay below 100 killometers per hour'],
    9:['Speed limit (120km/h)', 'You should stay below 120 killometers per hour'],
    10:['No passing', 'You should not pass'],
    11:['No passing veh over 3.5 tons', 'You should not pass if the vehicle is over 3.5 tons'],
    12:['Right-of-way at intersection', 'You should stay at the right of the road'],
    13:['Priority road', 'This is a priority road'],
    14:['Yield', 'You should yield to the next vehicle'],
    15:['Stop', 'You should stop the vehicle'],
    16:['No vehicles', 'No vehicles are allowed on this road'],
    17:['Veh > 3.5 tons prohibited', 'You should not pass if the vehicle is over 3.5 tons'],
    18:['No entry', 'You should not enter'],
    19:['General caution', 'General caution'],
    20:['Dangerous curve left', 'There is a dangerous curve to the left'],
    21:['Dangerous curve right', 'There is a dangerous curve to the right'],
    22:['Double curve', 'There is a double curve ahead'],
    23:['Bumpy road', 'There is a bumpy road ahead'],
    24:['Slippery road', 'There is a slippery road ahead'],
    25:['Road narrows on the right', 'The road is narrowing on the right'],
    26:['Road work', 'There is road work going on ahead'],
    27:['Traffic signals', 'There is a traffic signal ahead','Keep caution'],
    28:['Pedestrians', 'This is a pedestrian area'],
    29:['Children crossing', 'THIS IS A CHILDREN CROSSING AREA. \n Watch out for children \n Reduce Speed  \n Obey Any Signals From a Crossing Guard'],
    30:['Bicycles crossing', 'This is a bicycle area'],
    31:['Beware of ice/snow', 'There might be ice or snow ahead'],
    32:['Wild animals crossing', 'There might be wild animals ahead'],
    33:['End speed + passing limits', 'You should stay below 80 killometers per hour and not pass'],
    34:['Turn right ahead', 'You should turn right ahead'],
    35:['Turn left ahead', 'You should turn left ahead'],
    36:['Ahead only', 'You should not turn left or right ahead'],
    37:['Go straight or right', 'You should go straight or right and not turn left'],
    38:['Go straight or left', 'You should go straight or left and not turn right'],
    39:['Keep right', 'You should keep right and not turn left'],
    40:['Keep left', 'You should keep left and not turn right'],
    41:['Roundabout mandatory', 'You should go around the roundabout'],
    42:['End of no passing', 'You should not pass'],
    43:['End no passing vehicle with a weight greater than 3.5 tons', 'You should not pass if the vehicle is over 3.5 tons']
}

# Create class for the main window GUI
class Window(QWidget):
    def __init__(self):
        super().__init__() # The super() function returns the parent class of the class in which it is called.
        self.setWindowTitle('Traffic Sign Recognition')     # Set the title of the window
        self.setGeometry(200, 200, 1200, 800)               # Set the geometry of the new window
        self.setFixedSize(self.size())                      # Set the window to be of a fixed size
        self.UI()                                           # Call the UI function

    def UI(self): # Create the UI
        self.bg_label = QLabel(self)              # The background Image Label
        bg_image = QPixmap("C:\\Users\\HP\\Desktop\\FYP FINAL\\gui.jpg")          # Load the image
        bg_image = bg_image.scaled(1200, 800)     # Scale the image to window size
        self.bg_label.setPixmap(bg_image)         # Set it as the background of the label
        self.bg_label.setFixedSize(1200, 800)     # Set the size of the label to fit the entire window
        self.bg_label.show()                      # Show the label

        self.image_path = ''                                    # Create an empty string for the image path                # 
        self.add_image_button = QPushButton('ADD IMAGE', self)  # Create a button to add an image
        self.add_image_button.setToolTip('Add image')           # Set the tooltip for the button
        # Set button width to 15% of the window width and height to 10% of the window height 
        self.add_image_button.resize(int(self.width()*0.15), int(self.height()*0.10))   # Resize the button
        # Move button to 30% from the left and 30% from the top  
        self.add_image_button.move(int(self.width()*0.1375), int(self.height()*0.7))    # Move the button to the desired position
        self.add_image_button.setStyleSheet('QPushButton {background-color: yellow}')    # Set the button background color to yellow
        self.add_image_button.clicked.connect(self.add_image)                           # Connect the button to the add_image function
        
        self.classify_button = QPushButton('CLASSIFY', self)                            # Create a button to classify the image
        self.classify_button.setToolTip('Classify')                                     # Set the tooltip for the button
        self.classify_button.resize(int(self.width()*0.15), int(self.height()*0.10))    # Resize the button
        # Move button right of add image button
        self.classify_button.move(int(self.width()*0.4205), int(self.height()*0.7))     # Move the button to the desired position
        self.classify_button.setStyleSheet('QPushButton {background-color: yellow}')     # Set the button background color to yellow
        self.classify_button.clicked.connect(lambda: self.predict(self.image_path))     # Connect the button to the predict function
        # Disable button
        self.classify_button.setEnabled(False)                                          # Disable the button
        self.classify_button.show()                                                     # Show the button

        self.camera_button = QPushButton('CAMERA', self)                                # Create a button to classify the image
        self.camera_button.setToolTip('Use camera')                                   # Set the tooltip for the button
        self.camera_button.resize(int(self.width()*0.15), int(self.height()*0.10))    # Resize the button
        # Move button right of add image button
        self.camera_button.move(int(self.width()*0.7125), int(self.height()*0.7))     # Move the button to the desired position
        self.camera_button.setStyleSheet('QPushButton {background-color: yellow}')     # Set the button background color to yellow
        self.camera_button.clicked.connect(self.use_camera)     # Connect the button to the use camera function
        self.camera_button.show()

        self.image = QLabel(self)                                                           # Create a label to display the image
        self.image.resize(self.image.sizeHint())                                            # Resize the label
        # Move image to center of window 
        self.image.move(self.width()//2 - self.image.width()//2, int(self.height()*0.05))  # Move the label to the desired position
        # Set image dimensions to 300x300
        self.image.setFixedSize(300, 300)                                                   # Set the label to be of a fixed size   
        self.image.show()                                                                   # Show the label

        self.prob_label = QLabel(self)   
        self.prob_label.resize(self.image.sizeHint())                                        # Create a Label to to display the Probability 
        font_width = self.fontMetrics().width('Probability of accurate prediction: 100.00%') # Get the width of the font
        self.prob_label.move(int(self.width() * 0.7) , int(self.height()*0.1))               # Move the prob_label to the desired position
        self.prob_label.setFixedSize(300, 300)             
        self.prob_label.setText('No Image selected')                                        # Set the prob_label text to 'No Image selected'
        self.prob_label.setStyleSheet('QLabel {color: blue}')                               # Set the prob_label color to blue
        self.prob_label.show() 


        self.label = QLabel(self)                                                           # Create a label to display the image class 
        font_width = self.fontMetrics().width('No Image selected')                          # Get the width of the font
        self.label.move(self.width()//2 - self.label.width()//2, int(self.height()*0.43))   # Move the label to the desired position
        self.label.setText('No Image selected')                                             # Set the label text to 'No Image selected'
        # Set label color to Blue
        self.label.setStyleSheet('QLabel {color: blue; font-weight: bold}')                                    # Set the label color to blue
        self.label.show()                                                                   # Show the label

        self.description_box = QTextEdit(self)                                              # Create a text box to display the image class description
        self.description_box.setFixedHeight(int(self.height()*0.12))                        # Set the text box to be of a fixed height
        self.description_box.setFixedWidth(int(self.width()*0.4))                           # Set the text box to be of a fixed width           
        # Move the description box under the label
        self.description_box.move(self.width()//2 - self.description_box.width()//2, int(self.height()*0.49))   # Move the text box to the desired position
        self.description_box.setEnabled(False)                                              # Disable the text box
        # Set text box background color to gray      
        self.description_box.setStyleSheet('QTextEdit {background-color: gray;}')            # Set the text box background color to gray
        # Set text box font color to Red
        self.description_box.setStyleSheet('QTextEdit {color: red;}')                        # Set the text box font color to red
        self.show() # Show the window

        # Create a welcome window
        self.welcome_window = QMessageBox()
        self.welcome_window.setWindowTitle('Welcome')
        # Set text to show why following traffic signs are important
        self.welcome_window.setText('Traffic signs provide valuable information to drivers and other road users. They represent rules that are in place to keep you safe, and help to communicate messages to drivers and pedestrians that can maintain order and reduce accidents. Neglecting them can be dangerous.')
        # Set text color to red
        self.welcome_window.setStyleSheet('QLabel {color: red; font-weight: bold}')
        # Set window size to fit the text
        self.welcome_window.setFixedSize(self.welcome_window.sizeHint())
        # Show the window
        self.welcome_window.show()

    def add_image(self):
        # Create a file dialog to select an image
        self.image_path = QFileDialog.getOpenFileName(self, 'Open File')[0]
        # Reshape image to 300x300
        self.image.setPixmap(QPixmap(self.image_path))
        # Upscale image to 300x300
        self.image.setScaledContents(True)
        # Resize image to 300x300
        self.image.resize(300, 300) # Resize the label to the desired size
        self.image.move(self.width()//2 - self.image.width()//2, int(self.height()*0.05))  # Move the label to the desired position
        self.classify_button.setEnabled(True) # Enable the button
        self.image.show() # Show the image

    def _predict(self, image, model):
        x = model.predict(image)
        pred = np.argmax(x)
        y = round(x[0][np.argmax(x)], 2)
        prob = y * 100
        return pred, prob

    def predict(self, image_path):  # Function to predict the image class
        # Load the image
        image = keras.preprocessing.image.load_img(image_path, target_size=(30, 30))    
        # Convert the image to an array
        image = keras.preprocessing.image.img_to_array(image)  
        # Reshape the image to be of shape (1, 30, 30, 3)                         
        image = image.reshape(1, 30, 30, 3)

        # Get Prediction and Probablility using the detector model
        pred, prob = self._predict(image, detector_model)

        if pred == 1: # If the image is a traffic sign then
            # Get Prediction and Probablility using the classification model
            pred, prob = self._predict(image, model)
            # Get the class name from the index
            class_name = classes[pred+1] 

        else: # Otherwise use the class name from index 0
            class_name = classes[0]

        # Print the class name
        print(class_name[0])                  

        # Set the label text to the class name and justify the text
        self.label.setText(class_name[0])
        self.label.setAlignment(Qt.AlignCenter)
        self.label.adjustSize()

        # Set the probability label text to the class name and justify the text
        self.prob_label.setText(f"Probability of accurate prediction: {int(prob)}%")
        self.prob_label.setAlignment(Qt.AlignCenter)
        self.label.adjustSize()
       


        # Set the text of the description box to the class description and justify the text
        self.description_box.setText(class_name[1])
        self.description_box.setAlignment(Qt.AlignCenter)

        tts_engine.say(class_name[0] + ', ' + class_name[1])
        # Run without blocking by running it in a new thread
        Thread(target=tts_engine.runAndWait, daemon=True).start()

    def use_camera(self):
        proc = subprocess.Popen(['python', 'camera.py'])
        print('Camera started')
        proc.wait()
        self.image_path = 'image.jpg'
        self.image.setPixmap(QPixmap(self.image_path))
        self.image.setScaledContents(True)
        self.image.resize(300, 300) # Resize the label to the desired size
        self.image.move(self.width()//2 - self.image.width()//2, int(self.height()*0.05))  # Move the label to the desired position
        self.classify_button.setEnabled(True) # Enable the button
        self.image.show() # Show the image
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())
