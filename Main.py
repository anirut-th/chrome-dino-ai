import time
import numpy as np
import matplotlib.pyplot as plt
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
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
        # time.sleep(0.25)

    def duck(self):
        ActionChains(self._driver).key_down(Keys.ARROW_DOWN) \
            .click(self._driver.find_element_by_tag_name("body")) \
            .key_up(Keys.ARROW_DOWN) \
            .perform()
        # self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)
        # time.sleep(1)
    
    def do_action(self, action_id):
        if action_id == 1:
            self.jump()
        elif action_id == 2:
            self.duck()

    
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

def train():
    # model = keras.models.load_model('model/model.h5')
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(batch_input_shape=(1, 3)))
    model.add(keras.layers.Dense(10, activation='sigmoid'))
    model.add(keras.layers.Dense(10, activation='sigmoid'))
    model.add(keras.layers.Dense(3, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    game = Game()
    game.start()
    y = 0.99
    eps = 0.1
    decay_factor = 0.999
    r_avg_list = []
    num_episodes = 200
    for i in range(num_episodes):
        game.restart()
        qx, qy, qs, qr, qdone = game.get_state()
        s = np.array([qx, qy, qs])
        _s = np.expand_dims(s, axis=0)
        eps *= decay_factor
        print("Episode {} of {}".format(i + 1, num_episodes))
        done = False
        r_sum = 0
        while not done:
            print(game.get_state())

            if np.random.random() < eps:
                a = np.random.randint(0, 3)
            else:
                
                a = np.argmax(model.predict(_s))

            game.do_action(a)
            time.sleep(0.25)
            px, py, ps, pr, done = game.get_state()
            new_s = np.array([px, py, ps])
            _new_s = np.expand_dims(new_s, axis=0)
            target = pr + y * np.max(model.predict(_new_s))
            target_vec = model.predict(_s)[0]
            target_vec[a] = target
            model.fit(_s, target_vec.reshape(-1, 3), epochs=1, verbose=0)
            s = new_s
            r_sum += pr
        r_avg_list.append(r_sum / 1000)

    # game = Game()
    # game.start()
    # while True:
    #     state = game.get_state()
    #     print(state)
    #     if state[4]:
    #         break
    model.save('model/model.h5')
    game.end()

if __name__ == "__main__":
    train()
    
    # game = Game()
    # game.start()
    # while True:
    #     state = game.get_state()
    #     print(state)
    #     if state[4]:
    #         break
    # game.end()
    