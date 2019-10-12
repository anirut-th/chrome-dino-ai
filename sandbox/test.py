import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains


chrome_options = Options()
chrome_options.add_argument("disable-infobars")
driver = webdriver.Chrome(executable_path = "chromedriver.exe", chrome_options=chrome_options)
driver.set_window_position(x=-10,y=0)
driver.set_window_size(200, 300)
driver.get("chrome://dino/")
time.sleep(0.25)
driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
time.sleep(2)


print("done")
