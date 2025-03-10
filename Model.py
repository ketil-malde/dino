#!/usr/bin/python3

# Script to wrap yolo v5
# images must be placed in ./images/foobar.jpg (or png, etc)
# labels must be placed in ./labels/foobar.txt, format <classno> <x1> <y1> <x2> <y2> - in fraction of image

import os
import shutil
import pwd
import datetime

IMAGENAME='dino'

USERID=os.getuid()
GROUPID=os.getgid()
CWD=os.getcwd()
USERNAME=pwd.getpwuid(USERID).pw_name
RUNTIME='--gpus device=0' # ''

config = {
    'data_dir': 'data',
    'output_dir': 'runs',
    'arch': 'vit_small',
    'batch_size': 32,  # 40 probably fits in 12GB.
    'num_workers': 8,
    'epochs' : 100,
    'weights' : None
    }

def docker_run(args=''):
    os.system(f'docker run {RUNTIME} --ipc=host --rm --user {USERID}:{GROUPID} -v {CWD}:/project -it {USERNAME}-{IMAGENAME} {args}')

def docker_build(args=''):
    os.system(f'docker build --build-arg user={USERNAME} --build-arg uid={USERID} --build-arg gid={GROUPID} -t {USERNAME}-{IMAGENAME} {args}')

def docker_check(args=''):
    docker_run(f"python3 /usr/src/app/main_dino.py --help")
    
class Model:
    def __init__(self, conf, mypath):
        self.myconf = conf
        self.mypath = mypath

    def build(self):
        '''Build the docker instance'''
        print('Building docker...')
        docker_build(self.mypath)
        
    def train(self):
        '''Train the network'''
        now = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        outd = f"{self.myconf['output_dir']}/{now}"
        if self.myconf['weights'] is not None and os.path.isfile(self.myconf['weights']):
            print("*** Copying checkpoint {self.myconf.weights}")
            os.mkdir(outd)
            shutil.copy(self.myconf['weights'], outd)

        cmd = f"python3 /usr/src/app/main_dino.py --arch {self.myconf['arch']} --epochs {self.myconf['epochs']} --num_workers {self.myconf['num_workers']} --batch_size_per_gpu {self.myconf['batch_size']} --data_path {self.myconf['data_dir']} --output_dir {outd}"
        print('***',cmd)
        docker_run(cmd)

    def check(self):
        '''Verify that data is in place and that the output doesn't exist'''
        docker_check()

    def predict(self, wgths, target, output):
        '''Run a trained network on the data in target'''
        cmd = f"python3 /usr/src/app/video_generation.py --arch {self.myconf['arch']} --patch_size 16 --pretrained_weights {wgths} --input_path {target} --output_path {output}"
        print('***',cmd)
        docker_run(cmd)

    def test(self):
        '''Run tests'''
        pass

    def status(self):
        '''Print the current training status'''
        # check if docker exists
        # check if network is trained (and validation accuracy?)
        # check if data is present for training
        # check if test data is present
        # check if test output is present (and report)
        pass

if __name__ == '__main__':
    import argparse
    import sys
    
    if sys.argv[1] == 'build':
        docker_build('.')
    elif sys.argv[1] == 'check':
        docker_check()
    else:
        error('Usage: {sys.argv[0]} [check,status,train,predict] ...')

