import socket
import traceback
import time
import binascii

def utf8len(s):
    a = len(s.encode('utf-8'))
    b = "{:#010X}".format(a)
    return b[2:]
def s_x(a):
    c = "{:#06x}".format(a)
    return c[2:]

if __name__ == '__main__':
    tcp_server_addr = ('192.168.192.5', 19205)   # 端口号要按照需求进行调整  状态19204  控制19205 导航19206 配置19207
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        sock.connect(tcp_server_addr)
        print('connect success.')
    except Exception:
        print(traceback.format_exc())

    num = s_x(2002)   #编号
    testDict = '{"x":10.0,"y":3.0,"angle":0}'   #指令语句
    lenth = utf8len(testDict)
    testDict = binascii.b2a_hex(testDict.encode())
    testDict = testDict.decode()
    testDict = '5A010001'+str(lenth)+str(num)+'000000000000'+str(testDict)   #API报文
    testDict = testDict.upper().encode()
    sock.sendall(testDict)
    print(testDict)
    recvdata = sock.recv(2048).decode()


    print("小车响应状态:", recvdata)
    time.sleep(1)
    sock.close()