import pyaudio
import wave
def LuYin(Time,filename):
    CHUNK = 1024              #wav文件是由若干个CHUNK组成的，CHUNK我们就理解成数据包或者数据片段。
    FORMAT = pyaudio.paInt16  #这个参数后面写的pyaudio.paInt16表示我们使用量化位数 16位来进行录音。
    CHANNELS = 1              #代表的是声道，这里使用的单声道。
    RATE = 16000              # 采样率16k
    RECORD_SECONDS = Time     #采样时间
    WAVE_OUTPUT_FILENAME = filename   #输出文件名

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* 录音开始")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* 录音结束")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

import requests
import json
def Gettokent():
    baidu_server = "https://openapi.baidu.com/oauth/2.0/token?"
    grant_type = "client_credentials"
    #API Key
    client_id = "你的API Key"
    #Secret Key
    client_secret = "你的Secret Key"

    #拼url
    url = 'https://openapi.baidu.com/oauth/2.0/token?grant_type=client_credentials&client_id={}&client_secret={}'.format(client_id, client_secret)
    #print(url)
    #获取token
    res = requests.post(url)
    #print(res.text)
    token = json.loads(res.text)["access_token"]
    print(token)

import pyaudio
import wave
import requests
import json
import base64
import os
def BaiduYuYin(fileurl):
    try:
        RATE = "16000"                  #采样率16KHz
        FORMAT = "wav"                  #wav格式
        CUID = "wate_play"
        DEV_PID = "1536"                #无标点普通话
        token = '你的token'

        # 以字节格式读取文件之后进行编码
        with open(fileurl, "rb") as f:
            speech = base64.b64encode(f.read()).decode('utf8')

        size = os.path.getsize(fileurl)
        headers = {'Content-Type': 'application/json'}
        url = "https://vop.baidu.com/server_api"
        data = {
            "format": FORMAT,
            "rate": RATE,
            "dev_pid": DEV_PID,
            "speech": speech,
            "cuid": CUID,
            "len": size,
            "channel": 1,
            "token": token,
        }
        req = requests.post(url, json.dumps(data), headers)
        result = json.loads(req.text)
        return result["result"][0][:-1]
    except:
        return '识别不清'
