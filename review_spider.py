# encoding: utf-8

'''

@author: ZiqiLiu


@file: review_spider.py

@time: 2017/6/9 下午5:06

@desc:
'''
import requests
from bs4 import BeautifulSoup
import json

base_url = 'http://speechreview.in.naturali.io/prod/'

r = requests.get(base_url + 'get?limit=50&offset=0&deviceid=8FB56F7E4981B8D13D279C3C9BE5DEC5')
a = r.content
j = json.loads(a.decode())
for record in j['Detail']:
    download_url = base_url+'audio?key='+record['awskey']



