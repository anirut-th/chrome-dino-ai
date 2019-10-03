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
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    obst = cv2.imread('data/tree1.png', 0)
    obst_kp, obst_des = orb.detectAndCompute(obst, None)
    while(True):
        image = np.array(ImageGrab.grab(bbox=(0, 160, 480, 290)).convert('L'))
        image_kp, image_des = orb.detectAndCompute(image, None)

        matches = bf.match(obst_des, image_des)
        matches = sorted(matches, key = lambda x:x.distance)

        good_matches = matches[:10]

        src_pts = np.float32([obst_kp[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([image_kp[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # matchesMask = mask.ravel().tolist()
        # h,w = obst.shape[:2]
        # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

        # dst = cv2.perspectiveTransform(pts,M)
        # dst += (w, 0)  # adding offset

        # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
        #             singlePointColor = None,
        #             matchesMask = matchesMask, # draw only inliers
        #             flags = 2)

        img3 = cv2.drawMatches(obst, obst_kp, image, image_kp, good_matches, None)

        # Draw bounding box in Red
        # img3 = cv2.polylines(img3, [np.int32(dst)], True, (0,0,255),3, cv2.LINE_AA)
        cv2.imshow('window', img3)
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
    