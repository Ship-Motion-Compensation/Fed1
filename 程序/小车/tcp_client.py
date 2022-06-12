
import socket
import json
import time
import struct

PACK_FMT_STR = '!BBHLH6s'
IP = '192.168.192.5'
# Port = 19206


def packMasg(reqId, msgType, msg={}):
    msgLen = 0
    jsonStr = json.dumps(msg)
    if (msg != {}):
        msgLen = len(jsonStr)
    rawMsg = struct.pack(PACK_FMT_STR, 0x5A, 0x01, reqId, msgLen,msgType, b'\x00\x00\x00\x00\x00\x00')
    print("{:02X} {:02X} {:04X} {:08X} {:04X}"
    .format(0x5A, 0x01, reqId, msgLen, msgType))

    if (msg != {}):
        rawMsg += bytearray(jsonStr,'ascii')
        print(msg)

    return rawMsg

so = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
so.connect((IP, 19206))
so.settimeout(5)
test_msg = packMasg(1, 3051, {"source": "LM2", "id": "LM1", "task_id": "87654321"})
print("\n\nreq:")
print(' '.join('{:02X}'.format(x) for x in test_msg))
so.send(test_msg)

dataall = b''
# while True:
print('\n\n\n')
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
        print("{:02X} {:02X} {:04X} {:08X} {:04X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X}       length: {}"
              .format(header[0], header[1], header[2], header[3], header[4],
                      header[5][0], header[5][1], header[5][2], header[5][3], header[5][4], header[5][5],
                      header[3]))
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
        print(json.dumps(json.loads(data), indent=1))
        dataall += data
        print(' '.join('{:02X}'.format(x) for x in dataall))
except socket.timeout:
        print('timeout')

so.close()


while True:
    print("zhangshabi")
    so = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    so.connect((IP, 19204))
    so.settimeout(5)
    test_msg = packMasg(1, 1006, {})
    print("\n\nreq:")
    print(' '.join('{:02X}'.format(x) for x in test_msg))
    so.send(test_msg)

    dataall = b''
    # while True:
    print('\n\n\n')
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
        print("{:02X} {:02X} {:04X} {:08X} {:04X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X}       length: {}"
              .format(header[0], header[1], header[2], header[3], header[4],
                      header[5][0], header[5][1], header[5][2], header[5][3], header[5][4], header[5][5],
                      header[3]))
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
        print(json.dumps(json.loads(data), indent=1))
        dataall += data
        print(' '.join('{:02X}'.format(x) for x in dataall))
    except socket.timeout:
        print('timeout')
    time.sleep(2)
    so.close()

# so = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# so.connect((IP, 19206))
# so.settimeout(5)
# test_msg = packMasg(1,3001,{})
# print("\n\nreq:")
# print(' '.join('{:02X}'.format(x) for x in test_msg))
# so.send(test_msg)
#
# dataall = b''
# # while True:
# print('\n\n\n')
# try:
#     data = so.recv(16)
# except socket.timeout:
#     print('timeout')
#     so.close
# jsonDataLen = 0
# backReqNum = 0
# if(len(data) < 16):
#     print('pack head error')
#     print(data)
#     so.close()
# else:
#     header = struct.unpack(PACK_FMT_STR, data)
#     print("{:02X} {:02X} {:04X} {:08X} {:04X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X}       length: {}"
#     .format(header[0], header[1], header[2], header[3], header[4],
#     header[5][0], header[5][1], header[5][2], header[5][3], header[5][4], header[5][5],
#     header[3]))
#     jsonDataLen = header[3]
#     backReqNum = header[4]
# dataall += data
# data = b''
# readSize = 1024
# try:
#     while (jsonDataLen > 0):
#         recv = so.recv(readSize)
#         data += recv
#         jsonDataLen -= len(recv)
#         if jsonDataLen < readSize:
#             readSize = jsonDataLen
#     print(json.dumps(json.loads(data), indent=1))
#     dataall += data
#     print(' '.join('{:02X}'.format(x) for x in dataall))
# except socket.timeout:
#     print('timeout')
#
# so.close()