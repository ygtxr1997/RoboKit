import socket
import gradio as gr


# 定义一个函数，接受输入并返回相同的文本
def echo_text(input_text):
    return input_text

# 创建 Gradio 接口
iface = gr.Interface(fn=echo_text, inputs="text", outputs="text")

# Get Hostname
hostname = socket.gethostname()
domain_name = socket.getfqdn()


if 'dongxu-g' in hostname:
    gx = int(hostname[len('dongxu-g'):])
    server_port = 6000 + gx * 10
else:
    server_port = 1160

# 启动服务
iface.launch(server_name="0.0.0.0", server_port=server_port)
