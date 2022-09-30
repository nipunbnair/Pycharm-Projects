from pip._vendor.distlib.compat import raw_input
from socket import *

serverName = 'hostname'
serverPort = 12000
clientSocket = socket(AF_INET, SOCK_DGRAM)
message = raw_input('Input lowercase sentence:')
clientSocket.sendto(message.encode(encoding='utf-8'), serverName, serverPort)
modifiedMessage, serverAddress = clientSocket.recv(2048)
print('From Server: ', modifiedMessage.decode())
clientSocket.close()
