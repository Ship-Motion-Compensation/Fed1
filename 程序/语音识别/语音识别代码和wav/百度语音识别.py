#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import base64
import requests
import urllib.request
import time

class Speech_recognition():
    '''百度语音识别'''
    def key(self):
        # 获取token秘钥
        body = {
            "grant_type": "client_credentials",
            "client_id": "PltP1QOlXOLPB6GPQXQVXLAD",
            "client_secret": "HLsbsiBKElTlBEhOf4nWcxPb5GP5XF1V"
        }
        url = "https://aip.baidubce.com/oauth/2.0/token?"
        r = requests.post(url, data=body, verify=True, timeout=2)
        respond = json.loads(r.text)
        return respond["access_token"]

    def api(self, audio_data):
        with open(audio_data, 'rb') as fp:
            audio_data = fp.read()
        speech_data = base64.b64encode(audio_data).decode("utf-8")
        # 用Base64编码具有不可读性，需要解码后才能阅读
        speech_length = len(audio_data)
        post_data = {"format": "wav", "rate": 16000,
                     "channel": 1, "cuid": "123456python",
                     "token": self.key(), "speech": speech_data, "len": speech_length}

        json_data = json.dumps(post_data).encode("utf-8")
        json_length = len(json_data)

        try:
            req = urllib.request.Request("http://vop.baidu.com/server_api", data=json_data)
            req.add_header("Content-Type", "application/json")
            req.add_header("Content-Length", json_length)
            resp = urllib.request.urlopen(req, timeout=20)   # 合成时间不多，识别最耗时
            resp = resp.read()
            resp_data = json.loads(resp.decode("utf-8"))

        except:
            print('超时')

        if resp_data["err_no"] == 0:
            return resp_data["result"][0]
            # return resp_data["result"]  #返回识别出的文字
        else:
            return ''


if __name__ == '__main__':
    data = Speech_recognition().api("16k.wav")
    print("语音识别结果:", data)
    # input("--")