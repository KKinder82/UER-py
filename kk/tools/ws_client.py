# pip install websockets
# pip install asyncio

import websockets as ws
import asyncio
import threading

ws_url = "ws://gz.kaoxve.com:9099/socket"
async def sender(socket, buff):
    while True:
        if len(buff) < 1:
            return
        print("")
        print("     ---------------- send begin ----------------")
        data = buff.pop(0)
        print(data)
        await socket.send(data)
        print("     ----------------- send end -----------------")

async def heart(socket):
     await socket.send("##heart##")


async def receiver(socket):
    try:
        msg = await asyncio.wait_for(socket.recv(), timeout=1)
        print("")
        print("    >>>>>>>>>>>>>>>>>> recv Data <<<<<<<<<<<<<<<<")
        print(msg)
        print("    >>>>>>>>>>>>>>>> recv Data End <<<<<<<<<<<<<<<<")
    except asyncio.TimeoutError:
        return
    except ws.exceptions.ConnectionClosed:
        return


async def mLoop(buff):
    try:
        socket = await ws.connect(ws_url)
        while True:
            await sender(socket, buff)
            await receiver(socket)
            await heart(socket)
            await asyncio.sleep(1)
    except ws.exceptions.ConnectionClosed:
        print("ConnectionClosed")
        return

async def test():
    socket = await ws.connect(ws_url)
    # await socket.send("##heart##")
    # print("send heart")
    await socket.send("{}")
    msg = await socket.recv()
    print(msg)


def test2():
    socket = ws.connect(ws_url)
    socket.send("{}")
    msg = socket.recv()
    print(msg)


class ClientThread(threading.Thread):
    def __init__(self, name):
        super().__init__(name=name)
        self.buff = []

    def run(self):
        asyncio.run(mLoop(self.buff))


def main():
    client = ClientThread("wsclient")
    client.start()
    instr = input("请输入要发送的内容：")
    client.buff.append(instr)
    while True:
        instr = input()
        client.buff.append(instr)


if __name__ == "__main__":
    main()
    # test2()
    # asyncio.run(test())
    print("OK")
