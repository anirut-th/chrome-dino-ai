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
import csv
import cv2
import keyboard
from tools import find_clusters
from tools import cal_variance
from tools import compute_obj_size
from DQNAgent import DQNAgent
 
def screen_record():
    kernel = np.ones((3,3),np.uint8)
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,40)
    fontScale              = 1
    fontColor              = (255,0,0)
    lineType               = 2
    i = 1
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

        dino_img = np.copy(result[:-1, 0:80])
        player_pix = np.where(dino_img == 255)
        player_coor = list(zip(player_pix[0], player_pix[1]))
        player_y = 0
        if len(player_coor) > 0:
            _player_coor = np.asarray([[i[1], i[0]] for i in player_coor])
            y_player_coor = np.sort(_player_coor[:-1, 1])
            player_y = y_player_coor[0]
        
        road_img = np.copy(result[:-1, 95:480])
        x0, y0, x1, y1 = (990, 990, 999, 999)
        obst_pix = np.where(road_img == 255)
        obst_coor = list(zip(obst_pix[0], obst_pix[1]))
        if len(obst_coor) > 0:
            _obst_coor = np.asarray([[i[1], i[0]] for i in obst_coor])
            x_var = cal_variance(_obst_coor[:-1, 0])
            if x_var > 3000:
                clusters, labels = find_clusters(_obst_coor, 2)
                l = np.argmin(clusters[:-1, 0])
                m = np.where(labels == l)
                _obst_coor = _obst_coor[m]

            x_obst_coor = np.sort(_obst_coor[:-1, 0])
            x0 = x_obst_coor[0]
            x1 = x_obst_coor[-1]

            y_obst_coor = np.sort(_obst_coor[:-1, 1])
            y0 = y_obst_coor[0]
            y1 = y_obst_coor[-1]

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(image, (x0 + 95, y0), (x1 + 95, y1), (255,0,0), 2)
        cv2.line(image, (0, player_y), (60, player_y), (0,255,0), 2)
        cv2.imshow("window", image)
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
        self.actions = ActionChains(self._driver)
        self.current_action = ""


    def start(self):
        time.sleep(0.25)
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")
        #self._driver.refresh()
        #self.do_action(1)

    def is_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    def end(self):
        self._driver.close()

    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array) # the javascript object is of type array with score in the formate[1,0,0] which is 100.
        return int(score)

    def jump(self):
        if self.current_action == "":
            self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
        else:
            self.actions.key_up(self.current_action).perform()
            self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
            self.current_action == ""

    def duck(self):
        if self.current_action == "":
            self.actions.key_down(Keys.ARROW_DOWN).perform()
            self.current_action = Keys.ARROW_DOWN
        else:
            if self.current_action != Keys.ARROW_DOWN:
                self.actions.key_up(self.current_action).key_down(Keys.ARROW_DOWN).perform()
            else:
                self.actions.key_down(Keys.ARROW_DOWN).perform()
    def stand(self):
        if self.current_action != "":
            self.actions.key_up(self.current_action).perform()
            self.current_action = ""
    
    def do_action(self, action_id):
        if action_id == 1:
            self.jump()
        elif action_id == 2:
            self.duck()
        else:
            self.stand()

    
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

        dino_img = np.copy(result[:-1, 0:80])
        player_pix = np.where(dino_img == 255)
        player_coor = list(zip(player_pix[0], player_pix[1]))
        player_y = 0
        if len(player_coor) > 0:
            _player_coor = np.asarray([[i[1], i[0]] for i in player_coor])
            y_player_coor = np.sort(_player_coor[:-1, 1])
            player_y = y_player_coor[0]
        
        road_img = np.copy(result[:-1, 95:480])
        x0, y0, x1, y1 = (990, 990, 999, 999)
        obst_pix = np.where(road_img == 255)
        obst_coor = list(zip(obst_pix[0], obst_pix[1]))
        if len(obst_coor) > 0:
            _obst_coor = np.asarray([[i[1], i[0]] for i in obst_coor])
            x_var = cal_variance(_obst_coor[:-1, 0])
            if x_var > 3000:
                clusters, labels = find_clusters(_obst_coor, 2)
                l = np.argmin(clusters[:-1, 0])
                m = np.where(labels == l)
                _obst_coor = _obst_coor[m]

            x_obst_coor = np.sort(_obst_coor[:-1, 0])
            x0 = x_obst_coor[0]
            x1 = x_obst_coor[-1]

            y_obst_coor = np.sort(_obst_coor[:-1, 1])
            y0 = y_obst_coor[0]
            y1 = y_obst_coor[-1]
        
        width, height = compute_obj_size(x0, y0, x1, y1)
        distance = x0
        current_speed = self._driver.execute_script("return Runner.instance_.currentSpeed")
        current_score = self.get_score()
        is_crashed = self.is_crashed()
        return ([distance, width, height, 130 - player_y, current_speed], current_score, is_crashed)

def train(game):
    model = 0
    try:
        model = keras.models.load_model('models/model.h5')
    except:
        model = keras.Sequential()
        #model.add(keras.layers.Flatten(input_shape=(3)))
        model.add(keras.layers.Dense(5, input_shape=(5,)))
        model.add(keras.layers.Dense(20, activation='sigmoid'))
        model.add(keras.layers.Dense(20, activation='sigmoid'))
        model.add(keras.layers.Dense(3, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    game.start()
    y = 0.99
    eps = 0.5
    decay_factor = 0.999
    r_avg_list = []
    num_episodes = 10
    for i in range(num_episodes):
        game.restart()
        q_dist, q_width, q_height, q_player, q_spd, q_reward, qdone = game.get_state()
        s = np.array([q_dist, q_width, q_height, q_player, q_spd])
        _s = np.expand_dims(s, axis=0)
        eps *= decay_factor
        print("Episode {} of {}".format(i + 1, num_episodes))
        done = False
        r_sum = 0
        while not done:
            if np.random.random() < eps:
                a = np.random.randint(0, 3)
            else:
                a = np.argmax(model.predict(_s))

            game.do_action(a)
            #time.sleep(0.25)
            #print(game.get_state())
            p_dist, p_width, p_height, p_player, p_spd, p_reward, done = game.get_state()
            # if not done:
            #     new_s = np.array([ p_dist, p_width, p_height, p_player, p_spd])
            #     _new_s = np.expand_dims(new_s, axis=0)
            #     target = p_reward + y * np.max(model.predict(_new_s))
            #     target_vec = model.predict(_s)[0]
            #     target_vec[a] = target
            #     model.fit(_s, target_vec.reshape(-1, 3), epochs=1)
            #     s = new_s
            # else:
            #     target_vec = np.zeros(3) + 0.99
            #     target_vec[a] = 0
            #     model.fit(_s, target_vec.reshape(-1, 3), epochs=1)
            #     s = new_s
            if done:
                new_s = np.array([ p_dist, p_width, p_height, p_player, p_spd])
                target_vec = np.zeros(3) + 0.99
                target_vec[a] = 0
                model.fit(_s, target_vec.reshape(-1, 3), epochs=1)
                s = new_s

            r_sum += p_reward
        r_avg_list.append(r_sum / 1000)
    
    with open('test_result/result.csv', mode='w') as result_file:
        result_file = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(r_avg_list)):
            result_file.writerow([str(i + 1), r_avg_list[i]])

    model.save('models/model.h5')
    game.end()

def generateDataset(game):
    game.start()
    with open('temp/data.csv', mode='w') as result_file:
        result_file = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(10):
            game.restart()
            while not game.is_crashed():
                state = game.get_state()
                action = 0
                if keyboard.is_pressed('up'):
                    action = 1
                elif keyboard.is_pressed('down'):
                    action = 2
                result_file.writerow([state[0], state[1], state[2], action])
                time.sleep(0.1)
        
    game.end()

def train2(game):
    datas = []
    labels = []
    with open('temp/data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if len(row) > 0:
                datas.append([row[0], row[1], row[2]])
                l = np.zeros(3)
                index = int(row[3])
                l[index] = 0.99
                labels.append(l)
        print(f'Processed {line_count} lines.')
    datas = np.asarray(datas)
    labels = np.asarray(labels)
    print(labels.shape)
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(batch_input_shape=(1, 3)))
    model.add(keras.layers.Dense(10, activation='sigmoid'))
    model.add(keras.layers.Dense(10, activation='sigmoid'))
    model.add(keras.layers.Dense(3, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    model.fit(datas, labels, epochs=50)
    model.save('models/model.h5')
    game = Game()
    game.start()
    for i in range(10):
        game.restart()
        while not game.is_crashed():
            qx, qy, qs, qr, qdone = game.get_state()
            s = np.array([qx, qy, qs])
            _s = np.expand_dims(s, axis=0)
            a = np.argmax(model.predict(_s))
            game.do_action(a)

if __name__ == "__main__":
    episodes = 20
    agent = DQNAgent(5, 3)
    game = Game()
    game.start()
    total_reward = 0
    for e in range(episodes):
        game.restart()
        time.sleep(3)
        state, r, d = game.get_state()
        state = np.reshape(state, [1, 5])
        for time_t in range(50000):
            action = agent.act(state)
            game.do_action(action)
            next_state, reward, done = game.get_state()
            next_state = np.reshape(next_state, [1, 5])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(e, episodes, reward))
                total_reward = total_reward + reward
                break
            time.sleep(0.25)

        agent.replay(8)
    agent.save_model()
    game.end()
    print('average reward:', total_reward / episodes)
    #train(game)
    # while not game.is_crashed():
    #     print(game.get_state())
    # #train(game)
    # # chrome_options = Options()
    # # chrome_options.add_argument("disable-infobars")
    # # driver = webdriver.Chrome(executable_path = "chromedriver.exe", chrome_options=chrome_options)
    # # driver.set_window_position(x=-10,y=0)
    # # driver.set_window_size(200, 300)
    # # driver.get("chrome://dino/")
    # # screen_record()
    # game.end()
    