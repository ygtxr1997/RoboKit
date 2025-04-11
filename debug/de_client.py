import socket

def start_client(host='192.168.1.56', port=50003):
    # 创建一个客户端套接字
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # 连接到服务器
    client_socket.connect((host, port))

    # 发送消息
    message = "Hello, Server!"
    client_socket.sendall(message.encode())

    # 接收服务器的响应
    response = client_socket.recv(1024)
    print(f"Received from server: {response.decode()}")

    # 关闭连接
    client_socket.close()

if __name__ == "__main__":
    start_client()
