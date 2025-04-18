import socket
import select

def start_server(host='0.0.0.0', port=50003):
    # 创建一个 IPv4 TCP 套接字
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 设置 socket 选项，允许端口复用
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # 绑定到指定的 IP 和端口
    server_socket.bind((host, port))

    # 开始监听，最多允许 5 个连接
    server_socket.listen(5)
    print(f"Server started on {host}:{port}, waiting for connections...")

    # 设置非阻塞模式
    server_socket.setblocking(False)

    # 使用 select 来处理客户端连接
    inputs = [server_socket]  # 用于监听新连接的套接字
    outputs = []  # 当前没有需要写数据的套接字

    while True:
        # 使用 select.select() 来监听读写事件
        readable, writable, exceptional = select.select(inputs, outputs, inputs)

        # 处理所有准备好进行读取的套接字
        for sock in readable:
            if sock is server_socket:
                # 如果是 server_socket，表示有新客户端连接
                client_socket, client_address = server_socket.accept()
                print(f"Accepted connection from {client_address}")
                client_socket.setblocking(False)  # 设置客户端套接字为非阻塞模式
                inputs.append(client_socket)  # 将客户端套接字添加到 inputs 列表中
            else:
                # 处理现有连接上的数据
                data = sock.recv(1024)
                if data:
                    print(f"Received data: {data.decode()}")

                    # 发送响应给客户端
                    response = "Hello from server!"
                    sock.sendall(response.encode())
                else:
                    # 如果没有数据，表示客户端关闭了连接
                    print(f"Closing connection {sock.getpeername()}")
                    inputs.remove(sock)
                    sock.close()

        # 处理异常的套接字
        for sock in exceptional:
            print(f"Handling exceptional condition for {sock.getpeername()}")
            inputs.remove(sock)
            sock.close()

if __name__ == "__main__":
    start_server()
