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
        image = np.array(ImageGrab.grab(bbox=(89, 160, 480, 290)).convert('L'))
        bg_color = np.average(image)
        thresh_mode = 0
        if bg_color > 127:
            thresh_mode = cv2.THRESH_BINARY_INV
        else:
            thresh_mode = cv2.THRESH_BINARY

        ret, thresh = cv2.threshold(image, 127, 255, thresh_mode)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        result = cv2.cvtColor(opening, cv2.COLOR_GRAY2RGB)

        dist = np.where(result == 255)
        listOfCoordinates = list(zip(dist[0], dist[1]))
        if len(listOfCoordinates) > 0:
            listOfCoordinates.sort(key=lambda x: x[1])
            py, px = listOfCoordinates[0]
            lineThickness = 1
            cv2.line(result, (10, 20), (px, py), (0,255,0), lineThickness)

        cv2.imshow('window', result)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    
def GetScreenshot():
    im = ImageGrab.grab(bbox=(0, 170, 489, 300)).convert('L').filter(ImageFilter.FIND_EDGES)
    return im

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

if __name__ == "__main__":
    game = Game()
    game.start()
    screen_record()
    game.end()
    