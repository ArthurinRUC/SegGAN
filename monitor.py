import os
import sys
import time

cmd = 'python ~/code/train.py'


def gpu_info(gpu_index=1):
    info = os.popen(
        'nvidia-smi|grep %').read().split('\n')[gpu_index].split('|')
    power = int(info[1].split()[-3][:-1])
    memory = int(info[2].split('/')[0].strip()[:-3])
    usage = int(info[3].split(" ")[5][0:2])
    return power, memory, usage


i = 0
gpu_power, gpu_memory, gpu_usage = gpu_info()
while gpu_usage > 5:  # set waiting condition
    gpu_power, gpu_memory, gpu_usage = gpu_info()
    i = i % 5
    symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
    gpu_power_str = 'gpu power:%d W |' % gpu_power
    gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
    gpu_usage_str = 'gpu usage:%d |' % gpu_usage
    sys.stdout.write('\r' + gpu_usage_str + symbol)
    sys.stdout.flush()
    time.sleep(1)
    i += 1
print('\n' + cmd)
os.system(cmd)
