import clr

from sys import *
from sys.Collections.Generic import *
from sys.IO import *
from sys.Net import *
from sys.Net.Sockets import *
from sys.Runtime.InteropServices import *
from sys.Threading import *
from sys.Windows.Forms import *

class SeerMessageHead(object):
	def __init__(self):
 #保留 #保留 #保留 #保留 #保留 #保留
class SeerMessage(object):
	def __init__(self):

	def length(self):
		return self._data.Length + Marshal.SizeOf(self._head)

class MainForm(Form):
	def get__queueStop(self):

	def set__queueStop(self, value):

	_queueStop = property(fget=get__queueStop, fset=set__queueStop)

	#
	def bytesToStructure(bytesBuffer):
		if bytesBuffer.Length < Marshal.SizeOf(clr.GetClrType(T)):
			raise ArgumentException("size error")
		bufferHandler = Marshal.AllocHGlobal(bytesBuffer.Length)
		index = 0
		while index < bytesBuffer.Length:
			Marshal.WriteByte(bufferHandler, index, bytesBuffer[index])
			index += 1
		structObject = Marshal.PtrToStructure(bufferHandler, clr.GetClrType(T))
		Marshal.FreeHGlobal(bufferHandler)
		return structObject

	bytesToStructure = staticmethod(bytesToStructure)

	def seerMessageToBytes(self, msg):
		MessageBox.Show(msg.length().ToString())
		bytes = Array.CreateInstance(Byte, msg.length())
		structPtr = Marshal.AllocHGlobal(msg.length())
		Marshal.StructureToPtr(msg, structPtr, False)
		Marshal.Copy(structPtr, bytes, 0, msg.length())
		Marshal.FreeHGlobal(structPtr)
		return bytes

	def seerMessageHeadToBytes(self, msg):
		hsize = Marshal.SizeOf(msg)
		bytes = Array.CreateInstance(Byte, hsize)
		structPtr = Marshal.AllocHGlobal(hsize)
		Marshal.StructureToPtr(msg, structPtr, False)
		Marshal.Copy(structPtr, bytes, 0, hsize)
		Marshal.FreeHGlobal(structPtr)
		return bytes

	def hexStrTobyte(self, hexString):
		hexString = hexString.Replace(" ", "")
		if (hexString.Length % 2) != 0:
			hexString += " "
		returnBytes = Array.CreateInstance(Byte, hexString.Length / 2)
		i = 0
		while i < returnBytes.Length:
			returnBytes[i] = Convert.ToByte(hexString.Substring(i * 2, 2).Trim(), 16)
			i += 1
		return returnBytes

	def normalStrToHexByte(self, str):
		result = Array.CreateInstance(Byte, str.Length)
		buffer = System.Text.Encoding.UTF8.GetBytes(str)
		i = 0
		while i < buffer.Length:
			result[i] = Convert.ToByte(buffer[i].ToString("X2"), 16)
			i += 1
		return result

	def normalStrToHexStr(self, str):
		buffer = System.Text.Encoding.UTF8.GetBytes(str)
		result = str.Empty
		i = 0
		while i < buffer.Length:
			result += buffer[i].ToString("X2") + " "
			i += 1
		return result

	def __init__(self):
		self.InitializeComponent()

	def button_send_Click(self, sender, e):
		textBox_recv_data.Invoke(EventHandler())
		try:
			client = TcpClient(textBox_ip.Text.Trim(), Convert.ToInt32(textBox_port.Text.Trim()))
			if client.Connected:
				serverStream = client.GetStream()
				newmsg = SeerMessage()
				newmsg.head = self.bytesToStructure(self.hexStrTobyte(textBox_req_head.Text.Trim()))
				newmsg.data = self.normalStrToHexByte(textBox_req_data.Text.Trim())
				serverStream.Write(self.seerMessageHeadToBytes(newmsg.head), 0, Marshal.SizeOf(newmsg.head))
				serverStream.Write(newmsg.data, 0, newmsg.data.Length)
				serverStream.Flush()
				inStream = Array.CreateInstance(Byte, 16)
				while 16 != serverStream.Read(inStream, 0, 16):
					Thread.Sleep(20)
				recv_head = self.bytesToStructure(inStream)
				recvbyte = BitConverter.GetBytes(recv_head.length)
				Array.Reverse(recvbyte)
				dsize = BitConverter.ToUInt32(recvbyte, 0)
				bufferSize = 512
				datalist = List[Byte]()
				count = 0
				while True:
					buffer = Array.CreateInstance(Byte, bufferSize)
					readSize = serverStream.Read(buffer, 0, bufferSize)
					count += readSize
					datalist.AddRange(buffer)
					if count == dsize:
						break
					Thread.Sleep(10)
				textBox_recv_head.Text = BitConverter.ToString(self.seerMessageHeadToBytes(recv_head)).Replace("-", " ") #normalStrToHexStr(Encoding.UTF8.GetString(seerMessageHeadToBytes(recv_head)));
				str = System.Text.Encoding.UTF8.GetString(datalist.ToArray())
				textBox_recv_data.Invoke(EventHandler())
				client.Close()
		except SocketException, :
			textBox_recv_data.Invoke(EventHandler())
			MessageBox.Show("Connect Error!")
		except IOException, :
			textBox_recv_data.Invoke(EventHandler())
			MessageBox.Show("")
		finally:

	def textBox_req_data_TextChanged(self, sender, e):
		dsize = textBox_req_data.Text.Trim().Length
		head = self.bytesToStructure(self.hexStrTobyte(textBox_req_head.Text.Trim()))
		vv = self.hexStrTobyte(dsize.ToString("X8"))
		head.length = BitConverter.ToUInt32(vv, 0)
		textBox_req_head.Invoke(EventHandler())