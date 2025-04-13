## P2P Network

### 1. Install P2PSocket

Choose your version from the given [list](https://github.com/ygtxr1997/Wireboy.Socket.P2PSocket/releases/tag/3.3.2):

```shell
Client_linux_arm_v3.3.2.zip
Client_linux_x64_v3.3.2.zip
Client_osx_x64_v3.3.2.zip
Client_win_x64_v3.3.2.zip
Client_win_x86_v3.3.2.zip
Server_linux_arm_v3.3.2.zip
Server_linux_x64_v3.3.2.zip
Server_linux_x64_v3.3.2.zip
Server_win_x64_v3.3.2.zip
Server_win_x86_v3.3.2.zip
```

Next steps will take `Client_linux_x64_v3.3.2.zip` as an example.

```shell
cd /YourPathToRoboKit/thirdparty/
wget https://github.com/ygtxr1997/Wireboy.Socket.P2PSocket/releases/download/3.3.2/Client_linux_x64_v3.3.2.zip
unzip Client_linux_x64_v3.3.2.zip
cd Client_linux_x64/
```

There'll be a config file named `P2pSocket/Client.ini` under the extracted directory.


### 2. Edit the Config File

Write your own config file locally and copy its content into `P2pSocket/Client.ini`.
The file name of `P2pSocket/Client.ini` cannot be changed.


### 3. Start the P2P Program on Client

Note that both GPU inference server and robot local computer are taken as **Clients** in P2P connection.
Only the P2P network bridge server (owning a public IPV4 address) is the **Server**.

On Linux/Mac, run:

```shell
chmod +x P2PSocket.StartUp
./P2PSocket.StartUp
```

On Windows, double-click the `.exe` file:

```shell
P2PSocket.StartUp_Windows.exe
```

### 4. (Optional) Start the P2P Program on Server

As the **client user**, you do NOT need to set this.

```shell
chmod +x P2PSocket.StartUp  # on Linux/Mac
./P2PSocket.StartUp  
```

