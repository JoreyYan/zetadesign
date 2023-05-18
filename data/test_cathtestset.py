
import glob
import pickle
import pandas as pd
import numpy as np
import json
import biotite.structure.io.pdb.file as file
import biotite.application.dssp as dssp
import biotite.structure as struc
import tqdm
import wget
def read_cath_class(file):
    csvframe = pd.read_table(file, delim_whitespace=True, header=None)  # 读取txt文件的数字部分，无表头
    csvframe.columns = ['CATH domain name', 'Class', 'Architecture', 'Topology', 'Homologous superfamily',
                  'S35', 'S60', 'S95', 'S100',  'S100 sequence count', 'Domain length', 'resolution']
    return csvframe

def read_cath_s40_class(file):
    csvframe = pd.read_table(file, delim_whitespace=True, header=None)  # 读取txt文件的数字部分，无表头
    csvframe.columns = ['CATH domain name']
    return csvframe
def get_s40_info():
    file='./cath/cath-domain-list-v4_3_0.txt'
    s95=read_cath_class(file)
    file='./cath/cath-s40-noredu.txt'
    s40=read_cath_s40_class(file)
    intersected_df = pd.merge(s95, s40, on=['CATH domain name'], how='inner')
    filted_resolution=intersected_df[intersected_df.resolution<=3]
    filted_resolution_group=filted_resolution.groupby([ 'Class', 'Architecture', 'Topology'])

    topolopies=filted_resolution_group.groups.keys()
    topolopies_nums=len(topolopies)
    train_ratio=0.95
    test_id=np.sort(np.random.randint(low=0,high=topolopies_nums,size=int(topolopies_nums*(1-train_ratio))))

    test=[]
    train=[]

    test_topos=[]
    train_topos=[]


    i=0
    for label, option_course in filted_resolution_group:
        if i in test_id:
            test.append(option_course)
            test_topos.append(str(label))
        else:
            train.append(option_course)
            train_topos.append(str(label))
        i=i+1


    #write description
    description='there is ' +str(len(intersected_df)) +' domains in CATH-4.3-S40. '+ str(len(filted_resolution)) +' domains have resolutions <= 3.0.'  +'Which has  '+str(topolopies_nums)+' topolopies. '+'\n We take 95%//5% as trian and test set, ' +str(len(train))+' ' +str(len(test)) +' topolopies each. '


    test=pd.concat(test)
    train=pd.concat(train)
    description=description+str(len(train))+'//' +str(len(test)) +' domains each.'+'\n'
    print(description)
    train_domains=list(train['CATH domain name'])
    test_domains = list(test['CATH domain name'])
    wirtefile='CATH-S40-V4.3-Split.txt'
    split={
        'test':test_domains,
        'train': train_domains
    }
    split_topos={
        'test':test_topos,
        'train': train_topos
    }
    with open(wirtefile,'w') as f:
        f.writelines(description)
        f.writelines(50*'-')
        f.writelines('\n')
        f.writelines('\n')
        f.write(json.dumps(split))
        f.writelines('\n')
        f.writelines('\n')
        f.write(json.dumps(split_topos))
    print('\n')
    f.close()

def read_splits():
    splits='cath/CATH-S40-V4.3-Split.txt'
    with open(splits) as f:
        lines = f.readlines()
        domains_splits=json.loads(lines[4])
        test=domains_splits['test']
        train = domains_splits['train']

        return train,test


def get_test_topo():
    train, test=read_splits()
    file = './cath/cath-domain-list-v4_3_0.txt'
    s95 = read_cath_class(file)
    file = './cath/cath-s40-noredu.txt'
    s40 = read_cath_s40_class(file)
    intersected_df = pd.merge(s95, s40, on=['CATH domain name'], how='inner')
    filted_resolution = intersected_df[intersected_df.resolution <= 3]


    testpd=pd.DataFrame(test)
    testpd.columns = ['CATH domain name']

    testset = pd.merge(filted_resolution, testpd, on=['CATH domain name'], how='inner')

    testset_group=testset.groupby([ 'Class', 'Architecture','Topology']) #

    topos= {}

    for label, option_course in testset_group:
        test_domains = list(option_course['CATH domain name'])
        topos[str(label)]=test_domains

    wirtefile='cathcalssin_testset.txt'
    with open(wirtefile,'w') as f:
        f.writelines(json.dumps(topos))



def find_mulit():
    train, test=read_splits()
    file = './cath/cath-domain-list-v4_3_0.txt'
    s95 = read_cath_class(file)
    file = './cath/cath-s40-noredu.txt'
    s40 = read_cath_s40_class(file)
    intersected_df = pd.merge(s95, s40, on=['CATH domain name'], how='inner')
    filted_resolution = intersected_df[intersected_df.resolution <= 3]
    filted_resolution['pdbname']=filted_resolution['CATH domain name'].str[:4]
    filted_resolution['chainname']=filted_resolution['CATH domain name'].str[:5]

    chiangroups=filted_resolution.groupby([ 'chainname'])


    # chians=filted_resolution['chainname'].tolist()
    # chians=list(set(chians))
    #
    # wirtefile='cathchains.txt'
    # with open(wirtefile,'w') as f:
    #     f.writelines(json.dumps(chians))




    # multi=[]
    # single=[]
    # for label, option_course in chiangroups:
    #     if option_course.shape[0]==1:
    #         single.append(option_course)
    #     else:
    #         multi.append(option_course)
    #
    #
    testpd=pd.DataFrame(test)
    testpd.columns = ['CATH domain name']

    trainpd=pd.DataFrame(train)
    trainpd.columns = ['CATH domain name']

    testpd = pd.merge(filted_resolution, testpd, on=['CATH domain name'], how='inner')

    trainpd = pd.merge(filted_resolution, trainpd, on=['CATH domain name'], how='inner')

    traintest=pd.merge(testpd,trainpd, on=['chainname'], how='inner')
    x=traintest


    # testset['pdbname']=testset['CATH domain name'].str[:4]
    # testset['chainname']=testset['CATH domain name'].str[:5]
    #
    #
    #
    # pdbgroups=testset.groupby([ 'pdbname'])
    # chiangroups=testset.groupby([ 'chainname'])
    #




    # multi=[]
    # for label, option_course in groups:
    #     if option_course.shape[0]>1:
    #         multi.append(option_course)
    # x=glob.glob('/media/junyu/data/perotin/chain_set/cath43/1c3c*')
    #
    # print(len(multi))


def take_one_fromtopo():
    splits='cath/cathcalssin_testset.txt'
    with open(splits) as f:
        lines = f.readlines()
        domains_splits=json.loads(lines[0])
        for key in domains_splits:

            print(key,domains_splits[key])

take_one_fromtopo()
# # get_test_topo()
# take_one_fromtopo()