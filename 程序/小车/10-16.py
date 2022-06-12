import socket
hostname = '127.0.0.1' #设置主机名
port = 6666  #设置端口号 要确保这个端口号没有被使用，可以在cmd里面查看
addr = (hostname,port)
srv = socket.socket() #创建一个socket
srv.bind(addr)
srv.listen(5)
print("waitting connect")
while True:
 connect_socket,client_addr = srv.accept()
 print(client_addr)
 recevent = connect_socket.recv(1024)
 print(str(recevent,encoding='gbk'))
 connect_socket.send(bytes("你好，数据传输完成，这里是gaby-yan--server",encoding='gbk'))
 connect_socket.close()