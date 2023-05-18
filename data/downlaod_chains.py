import glob
import random

import tqdm
import wget
import json
import os
import threading

def download_singlechains():
    with open('cath/S95_filted.txt', 'r') as f:
        chians=f.readlines()[0]
    chiansx=json.loads(chians)
    url='http://cathdb.info/version/v4_3_0/api/rest/id/'
    paths='//home/jorey/CATH43_S95_filted/'
    # chiansx=chiansx[54000:]
    nums=len(chiansx)
    print(nums)
    val=[]
    threds=int(nums/500)
    for i in range(0,nums,500):
        val.append(chiansx[i:i+500])




    def download(inm):
        # random.shuffle(chiansx)
        for i in tqdm.tqdm(val[inm]):
            if not os.path.exists(paths+i+'.pdb'):
                file_name = wget.download(url+i+'.pdb',out=paths+i+'.pdb') # out=target_name

        print('finished ',inm)

    for i in range(threds):
        t = threading.Thread(target=download, args=( i,))
        t.start()

def down_multi_chains():
    dir='/home/junyu/下载/4039/extract/'
    subdir=os.listdir(dir)
    files=[]
    fs=[]
    for i in subdir:
        files=files+(os.listdir(dir+i+'/'))
        fs.append(os.listdir(dir+i+'/'))
    files=list(set(files))
    print(len(files))


    with open('cath/multi_chains.txt', 'r') as f:
        chians = f.readlines()[0]
    chiansx = chians.lower().split(',')
    url = 'http://cathdb.info/version/v4_3_0/api/rest/id/'
    paths = '/media/junyu/data/perotin/chain_set//mulit_chians/'

    nums = len(chiansx)
    val = []
    for i in range(0, nums, 50):
        val.append(chiansx[i:i + 50])

    def download(inm):
        # random.shuffle(chiansx)
        for i in tqdm.tqdm(val[inm]):
            if not os.path.exists(paths + i + '.pdb'):
                file_name = wget.download(url + i + '.pdb', out=paths + i + '.pdb')  # out=target_name

        print('finished ', inm)
        return

    for i in range(105):
        t = threading.Thread(target=download, args=(i,))
        t.start()

download_singlechains()
