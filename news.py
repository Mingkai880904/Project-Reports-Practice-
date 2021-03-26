# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 23:46:58 2021

@author: 沈明楷
"""

import requests
from bs4 import BeautifulSoup
response = requests.get(
    "https://www.ettoday.net/news/focus/%E7%A4%BE%E6%9C%83/")
soup = BeautifulSoup(response.text, "html.parser")
titles = soup.find_all("h3", itemprop="headline")
for title in titles:
    print(title.select_one("a").getText())