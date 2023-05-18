import glob
import pickle
import pandas as pd
import numpy as np
import json
import biotite.structure.io.pdb.file as file
import biotite.application.dssp as dssp
import biotite.structure as struc
import tqdm
def read_cath_class(file):
    csvframe = pd.read_table(file, delim_whitespace=True, header=None)  #
    csvframe.columns = ['CATH domain name', 'Class', 'Architecture', 'Topology', 'Homologous superfamily',
                  'S35', 'S60', 'S95', 'S100',  'S100 sequence count', 'Domain length', 'resolution']
    return csvframe
def read_cath_s40_class(file):
    csvframe = pd.read_table(file, delim_whitespace=True, header=None)  #
    csvframe.columns = ['CATH domain name']
    return csvframe


def filter_s95():
    file='./cath/cath-domain-list-S95-v4_3_0.txt'
    s95=read_cath_class(file)
    filted_resolution=s95[s95.resolution<=3]
    filted_list=filted_resolution['CATH domain name'].to_list()
    print(filted_resolution.shape)
    S95_filted='S95_filted.txt'
    with open(S95_filted,'w') as f:
        f.write(json.dumps(filted_list))
    print('\n')
    f.close()


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


def write_trainandtest(start=0,end=10000):
    train, test=read_splits()
    from data.data_transform import np_to_tensor_dict
    from model.data.data_pipeline import DataPipeline
    from data.data_transform import  atom37_to_torsion_angles

    datapipeline = DataPipeline()

    test_data=[]
    train_data=[]

    errortest_data=[]
    errortrain_data=[]

    pdb_dir='/media/junyu/data/perotin/chain_set/cath43/'
    pdb_fixed_dir='/home/junyu/pdb-tools/pdbtools/fixed/'
    jsonl_file='/media/junyu/data/perotin/chain_set/cath43_ALL.jsonl'
    np.set_printoptions(precision=5, suppress=True)
    with open(jsonl_file) as f:
        lines = f.readlines()[start:end]
        for line in tqdm.tqdm(lines):

            entry = json.loads(line)
            domain=entry['name']

            if domain in train:
                try:
                    pdb=pdb_dir+domain
                    x = datapipeline.process_pdb(pdb)
                except:
                    pdb = pdb_fixed_dir + 'fixed_' + domain + '.pdb'
                    x = datapipeline.process_pdb(pdb)
                entry['seq']=str(x['sequence'].tolist()[0]).split('\'')[1]
                xtensor = np_to_tensor_dict(x, ['aatype', 'all_atom_positions', 'all_atom_mask'])
                y = atom37_to_torsion_angles(xtensor)



                entry['torsion_angles_mask'] = np.round(y['torsion_angles_mask'].numpy(),8).tolist()
                entry['torsion_angles_sin_cos'] =(np.around(np.array(y['torsion_angles_sin_cos']),4)).tolist()
                entry['alt_torsion_angles_sin_cos'] = (np.around(np.array(y['alt_torsion_angles_sin_cos']),4)).tolist()
                train_data.append(entry)

            elif domain in test:
                try:
                    pdb = pdb_dir + domain
                    x = datapipeline.process_pdb(pdb)
                except:
                    pdb = pdb_fixed_dir + 'fixed_' + domain + '.pdb'
                    x = datapipeline.process_pdb(pdb)
                entry['seq']=str(x['sequence'].tolist()[0]).split('\'')[1]
                xtensor = np_to_tensor_dict(x, ['aatype', 'all_atom_positions', 'all_atom_mask'])
                y = atom37_to_torsion_angles(xtensor)



                entry['torsion_angles_mask'] = np.round(y['torsion_angles_mask'].numpy(),8).tolist()
                entry['torsion_angles_sin_cos'] = (np.around(np.array(y['torsion_angles_sin_cos']),4)).tolist()
                entry['alt_torsion_angles_sin_cos'] = (np.around(np.array(y['alt_torsion_angles_sin_cos']),4)).tolist()
                test_data.append(entry)

    print('\n error train are ',errortrain_data)
    print('\n')
    print('\n error test are ',errortest_data)
    print('\n')
    print('we want to write now')
    with open('test_n3w.jsonl', 'w') as f:
        for entry in test_data:
            f.write(json.dumps(entry) + '\n')
    f.close()
    with open('train_n3w.jsonl', 'w') as f:
        for entry in train_data:
            f.write(json.dumps(entry) + '\n')
    f.close()




def write_pdb_files():
    train, test=read_splits()
    from data.data_transform import np_to_tensor_dict
    from model.data.data_pipeline import DataPipeline
    from data.data_transform import  atom37_to_torsion_angles

    datapipeline = DataPipeline()

    test_data=[]
    train_data=[]

    errortest_data=[]
    errortrain_data=[]

    pdb_dir='/media/junyu/data/perotin/chain_set/cath43/'
    pdb_fixed_dir='/home/junyu/pdb-tools/pdbtools/fixed/'
    fixed_list=glob.glob(pdb_fixed_dir+'*.pdb')
    np.set_printoptions(precision=5, suppress=True)

    def get_entry(x,y):
        entry={}
        entry['seq']=x['sequence']#[0].decode('UTF-8')
        entry['aatype']=x['aatype']#.tolist()
        entry['domain_name']=x['domain_name']#[0].decode('UTF-8')
        entry['length']=len(x['sequence'][0].decode('UTF-8'))
        entry['residue_index']=x['residue_index']
        entry['backbone_atom_positions']=np.take(x['all_atom_positions'],[0,1,2,4],axis=1)#.tolist()
        entry['backbone_atom_mask']=np.take(x['all_atom_mask'],[0,1,2,4],axis=-1)#.tolist()
        if np.min(entry['backbone_atom_mask'])==0:
            print(entry['domain_name'])
        entry['seqmask']=seq_mask#.tolist()
        entry['torsion_angles_sin_cos']=y['torsion_angles_sin_cos']#.tolist()
        entry['alt_torsion_angles_sin_cos']=y['alt_torsion_angles_sin_cos']#.tolist()
        entry['torsion_angles_mask']=y['torsion_angles_mask']#.tolist()

        return entry



    for domain in tqdm.tqdm(train):
        if pdb_fixed_dir + 'fixed_' + domain + '.pdb' in fixed_list:
            pdb=pdb_fixed_dir + 'fixed_' + domain + '.pdb'
        else:
            pdb=pdb_dir+domain
        x = datapipeline.process_pdb(pdb)
        seq_mask=np.min(np.take(x['all_atom_mask'],[0,1,2,4],axis=-1),axis=-1)
        xtensor = np_to_tensor_dict(x, ['aatype', 'all_atom_positions', 'all_atom_mask'])
        y = atom37_to_torsion_angles(xtensor)

        entry=get_entry(x,y)
        train_data.append(entry)

    output = open('trainset.pkl', 'wb')

    # 写入到文件
    pickle.dump(train_data, output)
    output.close()

def read_pickel():
    test='testset.pkl'
    pkl_file = open(test, 'rb')
    testset=pickle.load(pkl_file)

    missrate=[]
    for i in testset:
        miss=np.sum(1-i['seqmask'])/np.sum(i['seqmask'])
        if miss>0.05:
            print(i['domain_name'])
        missrate.append(miss)
    print(np.max(missrate))

def write_pdb_files_fordesign(pdb_dir='/media/junyu/data/test1/'):

    from data.data_transform import np_to_tensor_dict
    from model.data.data_pipeline import DataPipeline
    from data.data_transform import  atom37_to_torsion_angles

    datapipeline = DataPipeline()


    design_data=[]





    fixed_list=glob.glob(pdb_dir+'*.pdb')

    np.set_printoptions(precision=5, suppress=True)

    def get_entry(x,y):
        entry={}
        entry['seq']=x['sequence']#[0].decode('UTF-8')
        entry['aatype']=x['aatype']#.tolist()
        entry['domain_name']=x['domain_name']#[0].decode('UTF-8')
        entry['length']=len(x['sequence'][0].decode('UTF-8'))
        entry['residue_index']=x['residue_index']
        entry['backbone_atom_positions']=np.take(x['all_atom_positions'],[0,1,2,4],axis=1)#.tolist()
        entry['backbone_atom_mask']=np.take(x['all_atom_mask'],[0,1,2,4],axis=-1)#.tolist()
        if np.min(entry['backbone_atom_mask'])==0:
            print(entry['domain_name'])
        entry['seqmask']=seq_mask#.tolist()
        entry['torsion_angles_sin_cos']=y['torsion_angles_sin_cos']#.tolist()
        entry['alt_torsion_angles_sin_cos']=y['alt_torsion_angles_sin_cos']#.tolist()
        entry['torsion_angles_mask']=y['torsion_angles_mask']#.tolist()

        return entry



    for domain in tqdm.tqdm(fixed_list):

        x = datapipeline.process_pdb(domain)
        seq_mask=np.min(np.take(x['all_atom_mask'],[0,1,2,4],axis=-1),axis=-1)
        xtensor = np_to_tensor_dict(x, ['aatype', 'all_atom_positions', 'all_atom_mask'])
        y = atom37_to_torsion_angles(xtensor)

        entry=get_entry(x,y)
        design_data.append(entry)

    output = open('Test1Design.pkl', 'wb')

    # 写入到文件
    pickle.dump(design_data, output)
    output.close()

if __name__ == '__main__':

    file='./cath/cath-s40-noredu.txt'
    read_cath_s40_class(file)
    get_s40_info()

