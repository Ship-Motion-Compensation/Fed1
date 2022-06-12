import socket

hostname = '127.0.0.1'
port = 6666
addr = (hostname, port)

clientsock = socket.socket()  ## 创建一个socket
clientsock.connect(addr)  # 建立连接

say = input("输入你想传送的消息：")
clientsock.send(bytes(say, encoding='gbk'))  # 发送消息
recvdata = clientsock.recv(1024)  # 接收消息 recvdata 是bytes形式的
print(str(recvdata, encoding='gbk'))  # 我们看不懂bytes，所以转化为 str
clientsock.close()