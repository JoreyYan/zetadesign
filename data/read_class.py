import sys
import os
import json
import glob
import tqdm
import shutil
splits='/home/oem/PDHS/data/cath/cathcalssin_testset.txt'


f=open(splits,'rb')
class_testset=json.loads(f.readlines()[0])


def mycopyfile(srcfile, dstpath):  # 复制函数
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.copy(srcfile, dstpath + fname)  # 复制文件
        print("copy %s -> %s" % (srcfile, dstpath + fname))


def read_splits(    splits='/home/junyu/PycharmProjects/PDHS/data/cath/CATH-S40-V4.3-Split.txt'):

    with open(splits) as f:
        lines = f.readlines()
        domains_splits=json.loads(lines[4])
        test=domains_splits['test']
        train = domains_splits['train']

        return train,test

def test():
    pdb_fixed_dir='//home/oem/pdb-tools/pdbtools/fixed/'
    pdb_dir='//home/oem/PDHS_DATA/cath43/dompdb/'
    fixed_list=glob.glob(pdb_fixed_dir+'*.pdb')

    choosen=[]

    for cluster,domains in class_testset.items():
        nums=len(domains)
        for domain in tqdm.tqdm(domains):

            if pdb_fixed_dir + 'fixed_' + domain + '.pdb' in fixed_list:
                pdb = pdb_fixed_dir + 'fixed_' + domain + '.pdb'
                continue
            else:
                pdb = pdb_dir + domain
                name = pdb.split('/')[-1].split('.')[0].split('_')[0]
                choosen.append([cluster,name])
                mycopyfile(pdb,'/home/oem/66class/nativefiles/')
                break

    f=open('/home/oem/66class/nativefiles//choosen_66clasee.txt','w')
    f.writelines(json.dumps(choosen))

    print(choosen)


    # train, test=read_splits()
    #
    # pdb_fixed_dir='//home/junyu/pdb-tools/pdbtools/fixed/'
    # pdb_dir='///media/junyu/data/perotin/chain_set/cath43/'
    # fixed_list=glob.glob(pdb_fixed_dir+'*.pdb')
    #
    # for domain in tqdm.tqdm(test):
    #
    #     if pdb_fixed_dir + 'fixed_' + domain + '.pdb'  in fixed_list:
    #         pdb =pdb_fixed_dir + 'fixed_' + domain + '.pdb'
    #         name = pdb.split('/')[-1].split('.')[0].split('_')[1]
    #     else:
    #         pdb =pdb_dir + domain
    #         name = pdb.split('/')[-1].split('.')[0].split('_')[0]

test()