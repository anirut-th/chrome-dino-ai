import time
import numpy as np
import matplotlib.pyplot as plt
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from PIL import ImageGrab
from PIL import ImageFilter
import tensorflow as tf
from tensorflow import keras

import cv2
 
def screen_record():
    kernel = np.ones((3,3),np.uint8)
    while(True):
        image = np.array(ImageGrab.grab(bbox=(0, 160, 480, 290)).convert('L'))
        

        bg_color = np.average(image)
        thresh_mode = 0
        if bg_color > 127:
            thresh_mode = cv2.THRESH_BINARY_INV
        else:
            thresh_mode = cv2.THRESH_BINARY

        ret, thresh = cv2.threshold(image, 127, 255, thresh_mode)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        result = cv2.cvtColor(opening, cv2.COLOR_GRAY2RGB)

        dino_img = np.copy(result[:-1, 0:89])
        road_img = np.copy(result[:-1, 89:480])

        x0 = 89
        y0 = 40
        x1 = 400
        y1 = 40
        
        lineThickness = 2
        obst = np.where(road_img == 255)
        listOfCoordinates = list(zip(obst[0], obst[1]))
        if len(listOfCoordinates) > 0:
            listOfCoordinates.sort(key=lambda x: x[1])
            y1, x1 = listOfCoordinates[0]

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        cv2.line(image, (x0, y0), (x1 + 89, y1), (0,255,0), lineThickness)
        cv2.imshow('window', image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

class Game:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        self._driver = webdriver.Chrome(executable_path = "chromedriver.exe", chrome_options=chrome_options)
        self._driver.set_window_position(x=-10,y=0)
        self._driver.set_window_size(200, 300)
        self._driver.get("chrome://dino/")

    def start(self):
        time.sleep(0.25)
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")

    def is_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    def end(self):
        self._driver.close()

    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array) # the javascript object is of type array with score in the formate[1,0,0] which is 100.
        return int(score)

    def jump(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
        time.sleep(0.25)

    def duck(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)
        time.sleep(0.25)
    
    def get_state(self):
        kernel = np.ones((3,3),np.uint8)
        image = np.array(ImageGrab.grab(bbox=(0, 160, 480, 290)).convert('L'))
        

        bg_color = np.average(image)
        thresh_mode = 0
        if bg_color > 127:
            thresh_mode = cv2.THRESH_BINARY_INV
        else:
            thresh_mode = cv2.THRESH_BINARY

        ret, thresh = cv2.threshold(image, 127, 255, thresh_mode)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        result = cv2.cvtColor(opening, cv2.COLOR_GRAY2RGB)

        dino_img = np.copy(result[:-1, 0:89])
        road_img = np.copy(result[:-1, 89:480])

        x0 = 89
        y0 = 40
        x1 = 999
        y1 = 999
        
        obst = np.where(road_img == 255)
        listOfCoordinates = list(zip(obst[0], obst[1]))
        if len(listOfCoordinates) > 0:
            listOfCoordinates.sort(key=lambda x: x[1])
            y1, x1 = listOfCoordinates[0]
        
        current_speed = self._driver.execute_script("return Runner.instance_.currentSpeed")
        current_score = self.get_score()
        is_crashed = self.is_crashed()
        return (x1, y1, current_speed, current_score, is_crashed)


if __name__ == "__main__":
    game = Game()
    game.start()
    while True:
        state = game.get_state()
        print(state)
        if state[4]:
            break
    game.end()
    