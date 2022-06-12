
import socket
import json
import time
import struct
import math

PACK_FMT_STR = '!BBHLH6s'
IP = '192.168.192.5'
# Port = 19206

def packMasg(reqId, msgType, msg={}):
    msgLen = 0
    jsonStr = json.dumps(msg)
    if (msg != {}):
        msgLen = len(jsonStr)
    rawMsg = struct.pack(PACK_FMT_STR, 0x5A, 0x01, reqId, msgLen,msgType, b'\x00\x00\x00\x00\x00\x00')
    # print("{:02X} {:02X} {:04X} {:08X} {:04X}"
    # .format(0x5A, 0x01, reqId, msgLen, msgType))

    if (msg != {}):
        rawMsg += bytearray(jsonStr,'ascii')
        # print(msg)

    return rawMsg

def move(Port, num, msg):
    # port：端口；num：编号；msg：指令语句
    so = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    so.connect((IP, Port))
    so.settimeout(5)
    test_msg = packMasg(1, num, msg)
    # print("\n\nreq:")
    # print(' '.join('{:02X}'.format(x) for x in test_msg))
    so.send(test_msg)

    dataall = b''
    # while True:
    # print('\n\n\n')
    try:
        data = so.recv(16)
    except socket.timeout:
        print('timeout')
        so.close
    jsonDataLen = 0
    backReqNum = 0
    if (len(data) < 16):
        print('pack head error')
        print(data)
        so.close()
    else:
        header = struct.unpack(PACK_FMT_STR, data)
        # print("{:02X} {:02X} {:04X} {:08X} {:04X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X}       length: {}"
        #       .format(header[0], header[1], header[2], header[3], header[4],
        #               header[5][0], header[5][1], header[5][2], header[5][3], header[5][4], header[5][5],
        #               header[3]))
        jsonDataLen = header[3]
        backReqNum = header[4]
    dataall += data
    data = b''
    readSize = 1024
    try:
        while (jsonDataLen > 0):
            recv = so.recv(readSize)
            data += recv
            jsonDataLen -= len(recv)
            if jsonDataLen < readSize:
                readSize = jsonDataLen
        # print(json.dumps(json.loads(data), indent=1))
        A = json.dumps(json.loads(data), indent=1)
        # print(A)
        # find1 = "MAC: 432"
        # print(A.find(find1))

        dataall += data
        # print(' '.join('{:02X}'.format(x) for x in dataall))
    except socket.timeout:
        print('timeout')

    so.close()
    return A

def sotp_move(M2):  # 检测到障碍物，暂停导航
    while True:
        A = move(19204, 1006, {})  # 查询状态信息
        find1 = '"blocked": true'
        if A.find(find1) != -1:
            move(19206, 13001, {})   #停止导航
            time.sleep(1)
            move(19206, 3056, {"angle": math.pi/3, "vw": math.pi/4})  # 转动
            time.sleep(4)
            move(19206, 3055, {"dist": 1, "vx": 0.3})  # 平动1米
            time.sleep(2)
            move(19206, 3056, {"angle": math.pi / 3, "vw": -math.pi / 4})  # 转动
            time.sleep(4)
            move(19206, 3055, {"dist": 1, "vx": 0.3})  # 平动1米
            time.sleep(2)
            break
        else:
            B = move(19204, 1004, {})
            find2 = '"current_station": ' + M2
            if B.find(find2) != -1:
                break
            time.sleep(1)

def nav(M1,M2):   #格式M1:"LM1",M1起始站点，M2终止站点
    while True:
        move(19206, 3051, {"source_id": M1, "id": M2, "task_id": "87654321"})  # 导航
        sotp_move(M2)
        B = move(19204, 1004, {})
        find2 = '"current_station": "M2"'
        if B.find(find2) != -1:
            break
if __name__ == '__main__':
    nav("LM1", "LM2")


