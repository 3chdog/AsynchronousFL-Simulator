import torch
import numpy as np
import client_instance
import subprocess
import asyncio

async def createClient(serverUrl=None, serverPort=None, delayTime=None):
    cmd = "python3 client_instance.py"
    if serverUrl != None: cmd += " -u {}".format(serverUrl)
    if serverPort != None: cmd += " -p {}".format(serverPort)
    if delayTime != None: cmd += " -d {}".format(delayTime)
    print(cmd)
    process = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate()

def task_release(num_clients):
    task_list = []
    for i in range(num_clients):
        task_list.append(createClient("127.0.0.1", 8081+i, 10))
    loop = asyncio.get_event_loop()
    commands = asyncio.gather(*task_list)
    reslt = loop.run_until_complete(commands)
    print(reslt)
    loop.close()




if __name__ == '__main__':
    np.random.seed(41)
    num_clients = 5
    server_url = "127.0.0.1"
    server_port = 8081

    task_release(num_clients)
