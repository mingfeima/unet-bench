import argparse
import subprocess
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from time import time

from unet2d import UNet
from unet3d import UNet3D

parser = argparse.ArgumentParser(description='UNet3D benchmark')
parser.add_argument('--no-cuda', action='store_true', default=False,
                   help='disable CUDA')
parser.add_argument('--batch-size', default=16, type=int,
                   help='batch size')
parser.add_argument('--in-channel', default=32, type=int,
                   help='input channel')
parser.add_argument('--prof', action='store_true', default=False,
                   help='enable autograd profiler')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

num_warmups = 1
num_iterations = 10

if args.cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    p = subprocess.check_output('nvidia-smi --query-gpu=name --format=csv', shell=True)
    device_name = str(p).split('\\n')[1]
else:
    p = subprocess.check_output('cat /proc/cpuinfo | grep name | head -n 1', shell = True)
    device_name = str(p).split(':')[1][:-3]

print('Running on device: %s' % (device_name))

def main():
    n = args.batch_size
    c = args.in_channel
    h = 128
    w = 128
    d = 128
    print('Model UNet, [N,C,H,W,D] = [%d,%d,%d,%d,%d]' % (n, c, h, w, d))

    data_ = torch.randn(n, c, h, w, d)
    target_ = torch.arange(1, n+1).long()
    #net = UNet(3, depth=5, merge_mode='concat')
    net = UNet3D(in_channel=args.in_channel, n_classes=6)
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    if args.cuda:
        data_, target_ = data_.cuda(), target_.cuda()
        net.cuda()

    net.eval()
    data, target = Variable(data_), Variable(target_)
    
    for i in range(num_warmups):
        optimizer.zero_grad()
        output = net(data)
        output.mean().backward()

    time_fwd, time_bwd, time_upt = 0, 0, 0

    for i in range(num_iterations):
        optimizer.zero_grad()
        t1 = time()
        output = net(data)
        t2 = time()
        output.mean().backward()
        t3 = time()
        optimizer.step()
        t4 = time()

        time_fwd += t2 - t1
        time_bwd += t3 - t2
        time_upt += t4 - t3
        print("iteration %d forward %10.2f ms, backward %10.2f ms" % (i, time_fwd*1000, time_bwd*1000))

    time_fwd_avg = time_fwd / num_iterations * 1000
    time_bwd_avg = time_bwd / num_iterations * 1000
    time_upt_avg = time_upt / num_iterations * 1000
    time_total = time_fwd_avg + time_bwd_avg

    print("%10s %10s %10s" % ('direction', "time(ms)", "imgs/sec"))
    print("%10s %10.2f %10.2f" % (':forward:', time_fwd_avg, n*1000/time_fwd_avg))
    print("%10s %10.2f" % (':backward:', time_bwd_avg))
    print("%10s %10.2f" % (':update:', time_upt_avg))
    print("%10s %10.2f %10.2f" % (':total:', time_total, n*1000/time_total))
                                     
if __name__ == '__main__':
    if args.prof:
        with torch.autograd.profiler.profile() as prof:
            main()
        f = open('profile.txt', 'w')
        f.write(prof.__str__())
    else:
        main()
