# encoding: utf-8

'''

@author: ZiqiLiu


@file: fetch_wave.py

@time: 2017/6/9 下午5:06

@desc:
'''
import requests
import json

device_id = '8FB56F7E4981B8D13D279C3C9BE5DEC5'
base_url = 'http://speechreview.in.naturali.io/prod/'


def fetch(id=device_id):
    r = requests.get(base_url + 'get?limit=50&offset=0&deviceid=' + device_id)
    a = r.content
    j = json.loads(a.decode())

    record = j['Detail'][0]
    download_url = base_url + 'audio/' + record['awskey']
    label = record['nires']
    print(download_url)
    wave = requests.get(download_url).content
    print(wave)
    return wave, label

