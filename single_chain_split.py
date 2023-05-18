import glob
import pickle
import pandas as pd
import numpy as np
import json
import biotite.structure.io.pdb.file as file
import biotite.application.dssp as dssp
import biotite.structure as struc
import tqdm
import random
import importlib
import torch
from model.np import residue_constants as rc
from data.data_transform import np_to_tensor_dict,tensor_to_np_dict
from model.data.data_pipeline import DataPipeline
from data.data_transform import  atom37_to_torsion_angles,atom37_to_frames,make_atom14_positions,make_atom14_masks,get_backbone_frames,make_atom14_bfactor
import argparse

def singles_chain_spilt():
    with open('data/cath/cathchains.txt', 'r') as f:
        chians=f.readlines()[0]
    chiansx=json.loads(chians)

    random.shuffle(chiansx)

    ratio=0.05

    test=chiansx[:int(ratio*len(chiansx))]
    train=chiansx[int(ratio*len(chiansx)):-1]

    dataset={'train':train,
    'test':test,
             }

    wirtefile = 'cath_chains_spilts.txt'
    with open(wirtefile, 'w') as f:
        f.writelines(json.dumps(dataset))


def read_splits(    splits='/home/jorey/pdhs/data/casp/CATH-S40-V4.3-Split.txt'):

    with open(splits) as f:
        lines = f.readlines()
        domains_splits=json.loads(lines[4])
        test=domains_splits['test']
        train = domains_splits['train']

        return train,test

def list_of_groups(list_info, per_list_len):
    '''
    :param list_info:
    :param per_list_len:
    :return:
    '''
    list_of_group = zip(*(iter(list_info),) *per_list_len)
    end_list = [list(i) for i in list_of_group] # i is a tuple
    count = len(list_info) % per_list_len
    end_list.append(list_info[-count:]) if count !=0 else end_list
    return end_list

def write_pdb_files_t0_pickl():
    # train, test=read_splits()
    from data.data_transform import np_to_tensor_dict
    from model.data.data_pipeline import DataPipeline
    from data.data_transform import  atom37_to_torsion_angles

    datapipeline = DataPipeline()

    test_data=[]
    train_data=[]

    errortest_data=[]
    errortrain_data=[]

    pdb_dir='/media/junyu/data/test1/singles/design/biochemtest/'

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
        # if np.min(entry['backbone_atom_mask'])==0:
        #     print(entry['domain_name'])
        entry['seqmask']=seq_mask#.tolist()
        entry['torsion_angles_sin_cos']=y['torsion_angles_sin_cos']#.tolist()
        entry['alt_torsion_angles_sin_cos']=y['alt_torsion_angles_sin_cos']#.tolist()
        entry['torsion_angles_mask']=y['torsion_angles_mask']#.tolist()

        return entry



    for domain in tqdm.tqdm(train):

        pdb=pdb_dir+domain+'.pdb'
        try:
            x = datapipeline.process_pdb(pdb)
        except:
            errortest_data.append(domain)
        seq_mask=np.min(np.take(x['all_atom_mask'],[0,1,2,4],axis=-1),axis=-1)
        xtensor = np_to_tensor_dict(x, ['aatype', 'all_atom_positions', 'all_atom_mask'])
        y = atom37_to_torsion_angles(xtensor)

        entry=get_entry(x,y)
        train_data.append(entry)

    output = open('DESIGNResult.pkl', 'wb')
    print('error:' ,errortest_data)
    # 写入到文件
    pickle.dump(train_data, output)
    output.close()


dssp_to_abc = {"I" : "c",
               "S" : "c",
               "H" : "a",
               "E" : "b",
               "G" : "c",
               "B" : "b",
               "T" : "c",
               "C" : "c",
               "-" : "-"}


def get_sse(pdbgzFile):
    import re
    def parse_PDB_biounits(x, sse, ssedssp, atoms=['N', 'CA', 'C'], chain=None):
        '''
        input:  x = PDB filename
                atoms = atoms to extract (optional)
        output: (length, atoms, coords=(x,y,z)), sequence
        '''

        alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
        states = len(alpha_1)
        alpha_3 = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                   'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'GAP']

        aa_1_N = {a: n for n, a in enumerate(alpha_1)}
        aa_3_N = {a: n for n, a in enumerate(alpha_3)}
        aa_N_1 = {n: a for n, a in enumerate(alpha_1)}
        aa_1_3 = {a: b for a, b in zip(alpha_1, alpha_3)}
        aa_3_1 = {b: a for a, b in zip(alpha_1, alpha_3)}

        def AA_to_N(x):
            # ["ARND"] -> [[0,1,2,3]]
            x = np.array(x);
            if x.ndim == 0: x = x[None]
            return [[aa_1_N.get(a, states - 1) for a in y] for y in x]

        def N_to_AA(x):
            # [[0,1,2,3]] -> ["ARND"]
            x = np.array(x);
            if x.ndim == 1: x = x[None]
            return ["".join([aa_N_1.get(a, "-") for a in y]) for y in x]

        xyz, seq, plddts, min_resn, max_resn = {}, {}, [], 1e6, -1e6

        pdbcontents = x.split('\n')[0]
        with open(pdbcontents) as f:
            pdbcontents = f.readlines()
        for line in pdbcontents:
            # line = line.decode("utf-8", "ignore").rstrip()

            if line[:6] == "HETATM" and line[17:17 + 3] == "MSE":
                line = line.replace("HETATM", "ATOM  ")
                line = line.replace("MSE", "MET")

            if line[:4] == "ATOM":
                ch = line[21:22]
                if ch == chain or chain is None or ch == ' ':
                    atom = line[12:12 + 4].strip()
                    resi = line[17:17 + 3]
                    resn = line[22:22 + 5].strip()
                    plddt = line[60:60 + 6].strip()

                    x, y, z = [float(line[i:(i + 8)]) for i in [30, 38, 46]]

                    if resn[-1].isalpha():
                        resa, resn = resn[-1], int(resn[:-1]) - 1  # in same pos ,use last atoms
                    else:
                        resa, resn = "_", int(resn) - 1
                    #         resn = int(resn)
                    if resn < min_resn:
                        min_resn = resn
                    if resn > max_resn:
                        max_resn = resn

                    if resn not in xyz:
                        xyz[resn] = {}
                    if resa not in xyz[resn]:
                        xyz[resn][resa] = {}
                    if resn not in seq:
                        seq[resn] = {}

                    if resa not in seq[resn]:
                        seq[resn][resa] = resi

                    if atom not in xyz[resn][resa]:
                        xyz[resn][resa][atom] = np.array([x, y, z])

        # convert to numpy arrays, fill in missing values
        seq_, xyz_, sse_, ssedssp_ = [], [], [], []
        dsspidx = 0
        sseidx = 0
        try:
            for resn in range(min_resn, max_resn + 1):
                if resn in seq:
                    for k in sorted(seq[resn]):
                        seq_.append(aa_3_N.get(seq[resn][k], 20))
                        try:
                            if 'CA' in xyz[resn][k]:
                                sse_.append(sse[sseidx])
                                sseidx = sseidx + 1
                            else:
                                sse_.append('-')
                        except:
                            print('error sse')


                else:
                    seq_.append(20)
                    sse_.append('-')

                misschianatom = False
                if resn in xyz:

                    for k in sorted(xyz[resn]):
                        for atom in atoms:
                            if atom in xyz[resn][k]:
                                xyz_.append(xyz[resn][k][
                                                atom])  # some will miss C and O ,but sse is normal,because sse just depend on CA
                            else:
                                xyz_.append(np.full(3, np.nan))
                                misschianatom = True
                        if misschianatom:
                            ssedssp_.append('-')
                            misschianatom = False
                        else:
                            try:
                                ssedssp_.append(
                                    ssedssp[dsspidx])  # if miss chain atom,xyz ,seq think is ok , but dssp miss this
                                dsspidx = dsspidx + 1
                            except:
                                print(dsspidx)


                else:
                    for atom in atoms:
                        xyz_.append(np.full(3, np.nan))
                    ssedssp_.append('-')

            return np.array(xyz_).reshape(-1, len(atoms), 3), N_to_AA(np.array(seq_)), np.array(sse_), np.array(
                ssedssp_)
        except TypeError:
            return 'no_chain', 'no_chain', 'no_chain'
    def parse_PDB(path_to_pdb, name, input_chain_list=None):
        """
        make sure every time just input 1 line
        """


        if input_chain_list:
            chain_alphabet = input_chain_list
        else:
            init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                             'S',
                             'T',
                             'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                             'm',
                             'n',
                             'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
            extra_alphabet = [str(item) for item in list(np.arange(300))]
            chain_alphabet = init_alphabet + extra_alphabet

        biounit_names = [path_to_pdb]
        for biounit in biounit_names:
            my_dict = {}
            s = 0
            concat_seq = ''

            for letter in chain_alphabet:

                PDBFile = file.PDBFile.read(biounit)
                array_stack = PDBFile.get_structure(altloc="all")

                sse1 = struc.annotate_sse(array_stack[0], chain_id=letter).tolist()
                if len(sse1) == 0:
                    sse1 = struc.annotate_sse(array_stack[0], chain_id='').tolist()
                ssedssp1 = dssp.DsspApp.annotate_sse(array_stack).tolist()
                xyz, seq, sse, ssedssp = parse_PDB_biounits(biounit, sse1, ssedssp1, atoms=['N', 'CA', 'C', 'O'],
                                                            chain=letter)

                return sse,ssedssp

    def parse_pdb_split_chain(pdbgzFile):

        with open(pdbgzFile) as f:
            lines = f.readlines()
            # pdbcontent = f.decode()

            pattern = re.compile('ATOM\s+\d+\s*\w+\s*[A-Z]{3,4}\s*(\w)\s*.+\n', re.MULTILINE)
            match = list(set(list(pattern.findall(lines[0]))))

        name = pdbgzFile.split('/')[-1]
        # for chain in match:
        # parse_PDB
        # match=[name[4]]
        # match=['A']
        pdb_data = parse_PDB(pdbgzFile, name, match)

        return pdb_data

    sse3,sse8=parse_pdb_split_chain(pdbgzFile)

    return sse3,sse8


def cut_length(entry,length):
    all_entrys=[]

    if entry['length']>length:
        nums=int(entry['length']/length)+1

        for i in range(nums):
            c_entry = {}
            for k,v in entry.items():
                if k!='length':
                    if (i+1)*length<=entry['length']:
                        c_entry[k]=v[i*length:(i+1)*length]
                    else:
                        c_entry[k] = v[entry['length']- length-1:-1]
                else:

                    c_entry['length'] = length


            all_entrys.append(c_entry)
    else:
        all_entrys.append(entry)
    return all_entrys


def centrility( X, mask, eps=1E-6):
    X=X.type(torch.float)
    mask=mask.type(torch.float)
    mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
    dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
    D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)
    D_max, _ = torch.max(D, -1, keepdim=True)
    D_adjust = D + (1. - mask_2D) * D_max


    cens=D_adjust<10
    cens=torch.sum(cens.squeeze(0),dim=-1).numpy()
    cens=(cens*mask.numpy()).astype(int).squeeze(0)
    #D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
    #return D_neighbors, E_idx
    return cens


def write_allatomspdb_files_to_pickl():
    train, test=read_splits(splits='/home/jorey/pdhs/data/casp/CATH-S40-V4.3-Split.txt')


    from data.data_transform import np_to_tensor_dict,tensor_to_np_dict
    from model.data.data_pipeline import DataPipeline
    from data.data_transform import  atom37_to_torsion_angles,atom37_to_frames,make_atom14_positions,make_atom14_masks,get_backbone_frames,make_atom14_bfactor

    datapipeline = DataPipeline()



    errortest_data=[]
    errortrain_data=[]

    pdb_fixed_dir='//home/junyu/pdb-tools/pdbtools/fixed/'
    pdb_dir='//home/jorey/pdhs_DATA/cath43/dompdb/'
    fixed_list=glob.glob(pdb_fixed_dir+'*.pdb')
    #train=glob.glob(pdb_dir+'*.pdb')

    # with open('cath/singles_chain_error.txt') as f:
    #     lines = f.readlines()
    #     domains_splits = json.loads(lines[0])
    # fixed_list=domains_splits['test']+domains_splits['train']



    np.set_printoptions(precision=8, suppress=True)

    def get_entry(x,y):
        entry={}
        #entry['seq']=x['sequence']#[0].decode('UTF-8')
        entry['aatype']=x['aatype']#.tolist()
        entry['sse3']=x['sse3']
        entry['sse8']=x['sse8']
        entry['cens']=x['cens']
        #entry['domain_name']=x['domain_name']#[0].decode('UTF-8')
        entry['length']=len(x['sequence'][0].decode('UTF-8'))
        entry['residue_index']=x['residue_index']

        entry['backbone_atom_positions']=np.take(x['all_atom_positions'],[0,1,2,4],axis=1)#.tolist()
        entry['backbone_atom_mask']=np.take(x['all_atom_mask'],[0,1,2,4],axis=-1)#.tolist()
        # if np.min(entry['backbone_atom_mask'])==0:
        #     print(entry['domain_name'])
        entry['seqmask']=seq_mask#.tolist()
        entry['b_factors']=y['atom14_atom_b_factors']
        entry['torsion_angles_sin_cos']=y['torsion_angles_sin_cos']#.tolist()
        entry['alt_torsion_angles_sin_cos']=y['alt_torsion_angles_sin_cos']#.tolist()
        entry['torsion_angles_mask']=y['torsion_angles_mask']#.tolist()
        entry['rigidgroups_gt_frames']=y['rigidgroups_gt_frames']#.tolist()
        entry['rigidgroups_gt_exists']=y['rigidgroups_gt_exists']#.tolist()

        entry['rigidgroups_alt_gt_frames']=y['rigidgroups_alt_gt_frames']#.tolist()
        entry['atom14_gt_positions']=y['atom14_gt_positions']#.tolist()
        entry['atom14_alt_gt_positions']=y['atom14_alt_gt_positions']#.tolist()
        entry['atom14_gt_exists']=y['atom14_gt_exists']#.tolist()
        entry['atom14_alt_gt_exists']=y['atom14_alt_gt_exists']#.tolist()
        entry['atom14_atom_exists']=y['atom14_atom_exists']#.tolist()
        entry['atom14_atom_is_ambiguous']=y['atom14_atom_is_ambiguous']#.tolist()

        ## recovert to 37
        entry['residx_atom37_to_atom14'] = y['residx_atom37_to_atom14']  # to add atoms
        entry['atom37_atom_exists'] = y['atom37_atom_exists']  #
        entry['all_atom_mask'] = y['all_atom_mask']  #
        entry['all_atom_positions'] = y['all_atom_positions']  #

        entry['residx_atom14_to_atom37'] = y['residx_atom14_to_atom37']  # to violence

        return entry



    datalist=list_of_groups(test,5000)




    for i in range(0,len(datalist)):
        data=datalist[i]
        train_data = []
        for domain in tqdm.tqdm(data):


            if pdb_fixed_dir +  domain.split('/')[-1]  in fixed_list:
                pdb =pdb_fixed_dir +  domain.split('/')[-1] #pdb_fixed_dir + 'fixed_' + domain + '.pdb'
            else:
                pdb =pdb_dir + domain#+'.pdb' #
            try:
                x,old_resx = datapipeline.process_pdb(pdb)
            except:
                errortrain_data.append(domain)

            old_res=old_resx-old_resx[0]

            sse3,sse8=get_sse(pdb)

            try:
                # sse3=np.take_along_axis(sse3,old_res,axis=0)
                sse8=np.take_along_axis(sse8,old_res,axis=0)
            except:
                try:
                    # sse3=np.delete(sse3,np.where(sse3=='-'))
                    sse8=np.delete(sse8,np.where(sse8=='-'))
                except:
                    print(old_res)



            seq_mask=np.min(np.take(x['all_atom_mask'],[0,1,2,4],axis=-1),axis=-1)

            # assert len(sse3)==len(seq_mask)
            if len(sse8)!=len(seq_mask):
                continue
            #
            sse3=[rc.sse3_order[dssp_to_abc[i]] for i in sse8 ]
            x['sse8']=np.asarray([rc.sse8_order[i] for i in sse8 ])
            x['sse3']=np.asarray(sse3)
            xtensor = np_to_tensor_dict(x, ['aatype', 'all_atom_positions', 'all_atom_mask','b_factors'])
            protein=atom37_to_frames(xtensor)
            protein=make_atom14_masks(protein)
            protein=make_atom14_positions(protein)
            protein=make_atom14_bfactor(protein)
            protein=get_backbone_frames(protein)
            protein = atom37_to_torsion_angles(protein)

            this_centrility = centrility(torch.as_tensor(protein['atom14_gt_positions'])[:, 3, :].unsqueeze(0),
                                         mask=torch.as_tensor(seq_mask).unsqueeze(0))  # Cb
            x['cens']=this_centrility
            del protein['all_atom_positions']
            del protein['all_atom_mask']



            protein=tensor_to_np_dict(protein,protein.keys())
            protein['domain_name']=x['domain_name']
            protein['residue_index']=x['residue_index']
            protein['length'] = np.asarray([len(x['aatype'])])

            for k,v in protein.items():
                if protein[k].dtype==np.float32:
                    protein[k]=protein[k].astype(np.float16)

            entry=get_entry(x,protein)
            entry=cut_length(entry,256)

            train_data=train_data+entry

        output = open('/home/jorey/pdhs/dataset/S40_allatoms_test_bfssecen_fix256_'+str(i)+'.pkl', 'wb')
        print('error:' ,errortest_data)
        # 写入到文件
        pickle.dump(train_data, output)
        output.close()

        del x
        del xtensor
        del protein
        del train_data
        import gc
        gc.collect()

def write_viotrainpdb_files_to_pickl():
    '''
    this is not sse part
    '''
    train, test=read_splits(splits='/home/jorey/pdhs/data/casp/CATH-S40-V4.3-Split.txt')

    from data.data_transform import np_to_tensor_dict,tensor_to_np_dict
    from model.data.data_pipeline import DataPipeline
    from data.data_transform import  atom37_to_torsion_angles,atom37_to_frames,make_atom14_positions,make_atom14_masks,get_backbone_frames

    datapipeline = DataPipeline()



    errortest_data=[]
    errortrain_data=[]

    pdb_fixed_dir='//home/jorey/pdb-tools/pdbtools/fixed/'
    pdb_dir='///home/jorey/cath43/'
    fixed_list=glob.glob(pdb_fixed_dir+'*')


    # with open('cath/singles_chain_error.txt') as f:
    #     lines = f.readlines()
    #     domains_splits = json.loads(lines[0])
    # fixed_list=domains_splits['test']+domains_splits['train']



    np.set_printoptions(precision=5, suppress=True)

    def get_entry(x,y):
        entry={}
        entry['seq']=x['sequence']#[0].decode('UTF-8')
        entry['aatype']=x['aatype']#.tolist()
        entry['domain_name']=x['domain_name']#[0].decode('UTF-8').split('_')[0]
        entry['length']=len(x['sequence'][0].decode('UTF-8'))
        entry['residue_index']=x['residue_index']
        entry['backbone_atom_positions']=np.take(x['all_atom_positions'],[0,1,2,4],axis=1)#.tolist()
        entry['backbone_atom_mask']=np.take(x['all_atom_mask'],[0,1,2,4],axis=-1)#.tolist()
        # if np.min(entry['backbone_atom_mask'])==0:
        #     print(entry['domain_name'])

        entry['sse3']=x['sse3']
        entry['sse8']=x['sse8']
        entry['cens']=x['cens']

        entry['b_factors']=y['b_factors']


        entry['seqmask']=seq_mask#.tolist()
        entry['torsion_angles_sin_cos']=y['torsion_angles_sin_cos']#.tolist()
        entry['alt_torsion_angles_sin_cos']=y['alt_torsion_angles_sin_cos']#.tolist()
        entry['torsion_angles_mask']=y['torsion_angles_mask']#.tolist()
        entry['rigidgroups_gt_frames']=y['rigidgroups_gt_frames']#.tolist()
        entry['rigidgroups_gt_exists']=y['rigidgroups_gt_exists']#.tolist()

        entry['rigidgroups_alt_gt_frames']=y['rigidgroups_alt_gt_frames']#.tolist()
        entry['atom14_gt_positions']=y['atom14_gt_positions']#.tolist()0
        entry['atom14_alt_gt_positions']=y['atom14_alt_gt_positions']#.tolist()
        entry['atom14_gt_exists']=y['atom14_gt_exists']#.tolist()
        entry['atom14_alt_gt_exists']=y['atom14_alt_gt_exists']#.tolist()
        entry['atom14_atom_exists']=y['atom14_atom_exists']#.tolist()
        entry['atom14_atom_is_ambiguous']=y['atom14_atom_is_ambiguous']#.tolist()

        entry['residx_atom37_to_atom14'] = y['residx_atom37_to_atom14']  # to add atoms
        entry['atom37_atom_exists'] = y['atom37_atom_exists']  #
        entry['all_atom_mask'] = y['all_atom_mask']  #
        entry['all_atom_positions'] = y['all_atom_positions']  #


        entry['residx_atom14_to_atom37'] = y['residx_atom14_to_atom37']  # to violence
        return entry



    datalist=list_of_groups(test,5000)




    for i in range(0,len(datalist)):
        train=datalist[i]
        train_data = []
        for domain in tqdm.tqdm(train):


            if pdb_fixed_dir + 'fixed_' + domain + ''  in fixed_list:
                pdb =pdb_fixed_dir + 'fixed_' + domain + ''
            else:
                pdb =pdb_dir + domain
            # try:
            #     x = datapipeline.process_pdb(pdb)
            # except:
            #     errortrain_data.append(domain)
            #
            # seq_mask=np.min(np.take(x['all_atom_mask'],[0,1,2,4],axis=-1),axis=-1)
            # xtensor = np_to_tensor_dict(x, ['aatype', 'all_atom_positions', 'all_atom_mask',])

            # pdb = domain
            try:
                x, old_resx = datapipeline.process_pdb(pdb)
            except:
                errortrain_data.append(domain)

            old_res = old_resx - old_resx[0]

            sse3, sse8 = get_sse(pdb)

            try:
                # sse3=np.take_along_axis(sse3,old_res,axis=0)
                sse8 = np.take_along_axis(sse8, old_res, axis=0)
            except:
                try:
                    # sse3=np.delete(sse3,np.where(sse3=='-'))
                    sse8 = np.delete(sse8, np.where(sse8 == '-'))
                except:
                    print(old_res)

            seq_mask = np.min(np.take(x['all_atom_mask'], [0, 1, 2, 4], axis=-1), axis=-1)

            if len(sse8) != len(seq_mask):
                continue

            sse3 = [rc.sse3_order[dssp_to_abc[i]] for i in sse8]
            x['sse8'] = np.asarray([rc.sse8_order[i] for i in sse8])
            x['sse3'] = np.asarray(sse3)
            xtensor = np_to_tensor_dict(x, ['aatype', 'all_atom_positions', 'all_atom_mask', 'b_factors'])



            protein=atom37_to_frames(xtensor)
            protein=make_atom14_masks(protein)
            protein=make_atom14_positions(protein)
            protein=get_backbone_frames(protein)
            protein = atom37_to_torsion_angles(protein)



            this_centrility = centrility(torch.as_tensor(protein['atom14_gt_positions'])[:, 3, :].unsqueeze(0),
                                         mask=torch.as_tensor(seq_mask).unsqueeze(0))  # Cb
            x['cens']=this_centrility

            protein=tensor_to_np_dict(protein,protein.keys())
            protein['domain_name']=x['domain_name']
            protein['residue_index']=x['residue_index']
            protein['length'] = np.asarray([len(x['aatype'])])

            for k,v in protein.items():
                if protein[k].dtype==np.float32:
                    protein[k]=protein[k].astype(np.float16)

            entry=get_entry(x,protein)
            train_data.append(entry)

        output = open('dataset/S40_repacker_vio_test_'+str(i)+'.pkl', 'wb')
        print('error:' ,errortest_data)
        # 写入到文件
        pickle.dump(train_data, output)
        output.close()

        del x
        del xtensor
        del protein
        del train_data
        import gc
        gc.collect()

def write_design_pdb_files_to_pickl(pdb_dir,ouputdir):


    datapipeline = DataPipeline()

    test_data=[]
    train_data=[]

    errortest_data=[]
    errortrain_data=[]


    fixed_list=glob.glob(pdb_dir+'**.pdb') #/home/junyu/下载/CASP14

    np.set_printoptions(precision=5, suppress=True)

    def get_entry(x,y):
        entry={}
        entry['seq']=x['sequence']#[0].decode('UTF-8')
        entry['aatype']=x['aatype']#.tolist()

        entry['sse3']=x['sse3']
        entry['sse8']=x['sse8']
        entry['cens']=x['cens']


        entry['domain_name']=x['domain_name']#[0].decode('UTF-8').split('_')[0]
        entry['length']=len(x['sequence'][0].decode('UTF-8'))
        entry['residue_index']=x['residue_index']
        entry['backbone_atom_positions']=np.take(x['all_atom_positions'],[0,1,2,4],axis=1)#.tolist()
        entry['backbone_atom_mask']=np.take(x['all_atom_mask'],[0,1,2,4],axis=-1)#.tolist()
        # if np.min(entry['backbone_atom_mask'])==0:
        #     print(entry['domain_name'])
        entry['seqmask']=seq_mask#.tolist()
        entry['b_factors']=y['atom14_atom_b_factors']
        entry['torsion_angles_sin_cos']=y['torsion_angles_sin_cos']#.tolist()
        entry['alt_torsion_angles_sin_cos']=y['alt_torsion_angles_sin_cos']#.tolist()
        entry['torsion_angles_mask']=y['torsion_angles_mask']#.tolist()
        entry['rigidgroups_gt_frames']=y['rigidgroups_gt_frames']#.tolist()
        entry['rigidgroups_gt_exists']=y['rigidgroups_gt_exists']#.tolist()

        entry['rigidgroups_alt_gt_frames']=y['rigidgroups_alt_gt_frames']#.tolist()
        entry['atom14_gt_positions']=y['atom14_gt_positions']#.tolist()
        entry['atom14_alt_gt_positions']=y['atom14_alt_gt_positions']#.tolist()
        entry['atom14_gt_exists']=y['atom14_gt_exists']#.tolist()
        entry['atom14_alt_gt_exists']=y['atom14_alt_gt_exists']#.tolist()
        entry['atom14_atom_exists']=y['atom14_atom_exists']#.tolist()
        entry['atom14_atom_is_ambiguous']=y['atom14_atom_is_ambiguous']#.tolist()

        entry['residx_atom37_to_atom14'] = y['residx_atom37_to_atom14']  # to add atoms
        entry['atom37_atom_exists'] = y['atom37_atom_exists']  #
        entry['all_atom_mask'] = y['all_atom_mask']  #
        entry['all_atom_positions'] = y['all_atom_positions']  #


        entry['residx_atom14_to_atom37'] = y['residx_atom14_to_atom37']  # to violence
        return entry


    name=[]
    for domain in tqdm.tqdm(fixed_list):

            pdb=domain
            try:
                x, old_resx = datapipeline.process_pdb(pdb)
            except:
                errortrain_data.append(domain)

            old_res = old_resx - old_resx[0]

            sse3, sse8 = get_sse(pdb)

            try:
                # sse3=np.take_along_axis(sse3,old_res,axis=0)
                sse8 = np.take_along_axis(sse8, old_res, axis=0)
            except:
                try:
                    # sse3=np.delete(sse3,np.where(sse3=='-'))
                    sse8 = np.delete(sse8, np.where(sse8 == '-'))
                except:
                    print(old_res)

            seq_mask = np.min(np.take(x['all_atom_mask'], [0, 1, 2, 4], axis=-1), axis=-1)


            if len(sse8) != len(seq_mask):
                continue

            sse3=[rc.sse3_order[dssp_to_abc[i]] for i in sse8 ]
            x['sse8']=np.asarray([rc.sse8_order[i] for i in sse8 ])
            x['sse3']=np.asarray(sse3)
            xtensor = np_to_tensor_dict(x, ['aatype', 'all_atom_positions', 'all_atom_mask','b_factors'])






            protein=atom37_to_frames(xtensor)
            protein=make_atom14_masks(protein)
            protein=make_atom14_positions(protein)
            protein=make_atom14_bfactor(protein)
            protein=get_backbone_frames(protein)
            protein = atom37_to_torsion_angles(protein)

            this_centrility = centrility(torch.as_tensor(protein['atom14_gt_positions'])[:, 3, :].unsqueeze(0),
                                         mask=torch.as_tensor(seq_mask).unsqueeze(0))  # Cb
            x['cens']=this_centrility

            # del protein['all_atom_positions']
            # del protein['all_atom_mask']

            name.append(x['domain_name'])
            x['residue_index']=old_resx
            protein=tensor_to_np_dict(protein,protein.keys())
            protein['domain_name']=x['domain_name']
            protein['residue_index']=x['residue_index']
            protein['length'] = np.asarray([len(x['aatype'])])

            for k,v in protein.items():
                if protein[k].dtype==np.float32:
                    protein[k]=protein[k].astype(np.float16)

            entry=get_entry(x,protein)
            entry=[entry]

            train_data=train_data+entry





    print('error:', errortest_data)

    output = open(ouputdir+'/processed_pdb.pkl', 'wb')

    # 写入到文件
    pickle.dump(train_data, output)
    output.close()


def write_fbb_pdb_files_to_pickl():
    train, test=read_splits(splits='/home/jorey/pdhs/data/casp/CATH-S40-V4.3-Split.txt')


    from data.data_transform import np_to_tensor_dict,tensor_to_np_dict
    from model.data.data_pipeline import DataPipeline
    from data.data_transform import  atom37_to_torsion_angles,atom37_to_frames,make_atom14_positions,make_atom14_masks,get_backbone_frames,make_atom14_bfactor

    datapipeline = DataPipeline()



    errortest_data=[]
    errortrain_data=[]

    pdb_fixed_dir='//home/jorey/pdb-tools/pdbtools/S40/'
    pdb_dir='//home/jorey/CATH43_S40_filted/'
    fixed_list=glob.glob(pdb_fixed_dir+'*.pdb')
    train=glob.glob(pdb_dir+'*.pdb')

    # with open('cath/singles_chain_error.txt') as f:
    #     lines = f.readlines()
    #     domains_splits = json.loads(lines[0])
    # fixed_list=domains_splits['test']+domains_splits['train']



    np.set_printoptions(precision=8, suppress=True)

    def get_entry(x,y):
        entry={}
        #entry['seq']=x['sequence']#[0].decode('UTF-8')
        entry['aatype']=x['aatype']#.tolist()
        # entry['sse3']=x['sse3']
        # entry['sse8']=x['sse8']
        # entry['cens']=x['cens']
        #entry['domain_name']=x['domain_name']#[0].decode('UTF-8')
        entry['length']=len(x['sequence'][0].decode('UTF-8'))
        entry['residue_index']=x['residue_index']

        entry['backbone_atom_positions']=np.take(x['all_atom_positions'],[0,1,2,4],axis=1)#.tolist()
        entry['backbone_atom_mask']=np.take(x['all_atom_mask'],[0,1,2,4],axis=-1)#.tolist()
        # if np.min(entry['backbone_atom_mask'])==0:
        #     print(entry['domain_name'])
        entry['seqmask']=seq_mask#.tolist()
        entry['b_factors']=y['atom14_atom_b_factors']
        entry['torsion_angles_sin_cos']=y['torsion_angles_sin_cos']#.tolist()
        entry['alt_torsion_angles_sin_cos']=y['alt_torsion_angles_sin_cos']#.tolist()
        entry['torsion_angles_mask']=y['torsion_angles_mask']#.tolist()
        entry['rigidgroups_gt_frames']=y['rigidgroups_gt_frames']#.tolist()
        entry['rigidgroups_gt_exists']=y['rigidgroups_gt_exists']#.tolist()

        # entry['rigidgroups_alt_gt_frames']=y['rigidgroups_alt_gt_frames']#.tolist()
        # entry['atom14_gt_positions']=y['atom14_gt_positions']#.tolist()
        # entry['atom14_alt_gt_positions']=y['atom14_alt_gt_positions']#.tolist()
        # entry['atom14_gt_exists']=y['atom14_gt_exists']#.tolist()
        # entry['atom14_alt_gt_exists']=y['atom14_alt_gt_exists']#.tolist()
        # entry['atom14_atom_exists']=y['atom14_atom_exists']#.tolist()
        # entry['atom14_atom_is_ambiguous']=y['atom14_atom_is_ambiguous']#.tolist()

        # ## recovert to 37
        # entry['residx_atom37_to_atom14'] = y['residx_atom37_to_atom14']  # to add atoms
        # entry['atom37_atom_exists'] = y['atom37_atom_exists']  #
        # entry['all_atom_mask'] = y['all_atom_mask']  #
        # entry['all_atom_positions'] = y['all_atom_positions']  #
        #
        # entry['residx_atom14_to_atom37'] = y['residx_atom14_to_atom37']  # to violence

        return entry



    datalist=list_of_groups(train,5000)




    for i in range(0,len(datalist)):
        data=datalist[i]
        train_data = []
        for domain in tqdm.tqdm(data):

            X=domain.split('/')[-1]
            if pdb_fixed_dir +  domain.split('/')[-1]  in fixed_list:
                pdb =pdb_fixed_dir +  domain.split('/')[-1] #pdb_fixed_dir + 'fixed_' + domain + '.pdb'
            else:
                pdb =pdb_dir + domain.split('/')[-1]#+'.pdb' #
            try:
                x,old_resx = datapipeline.process_pdb(pdb)
            except:
                errortrain_data.append(domain)

            old_res=old_resx-old_resx[0]
            x['residue_index']=old_res
            # sse3,sse8=get_sse(pdb)
            #
            # try:
            #     # sse3=np.take_along_axis(sse3,old_res,axis=0)
            #     sse8=np.take_along_axis(sse8,old_res,axis=0)
            # except:
            #     try:
            #         # sse3=np.delete(sse3,np.where(sse3=='-'))
            #         sse8=np.delete(sse8,np.where(sse8=='-'))
            #     except:
            #         print(old_res)



            seq_mask=np.min(np.take(x['all_atom_mask'],[0,1,2,4],axis=-1),axis=-1)

            # # assert len(sse3)==len(seq_mask)
            # if len(sse8)!=len(seq_mask):
            #     continue
            # #
            # sse3=[rc.sse3_order[dssp_to_abc[i]] for i in sse8 ]
            # x['sse8']=np.asarray([rc.sse8_order[i] for i in sse8 ])
            # x['sse3']=np.asarray(sse3)
            xtensor = np_to_tensor_dict(x, ['aatype', 'all_atom_positions', 'all_atom_mask','b_factors'])
            protein=atom37_to_frames(xtensor)
            protein=make_atom14_masks(protein)
            protein=make_atom14_positions(protein)
            protein=make_atom14_bfactor(protein)
            protein=get_backbone_frames(protein)
            protein = atom37_to_torsion_angles(protein)

            # this_centrility = centrility(torch.as_tensor(protein['atom14_gt_positions'])[:, 3, :].unsqueeze(0),
            #                              mask=torch.as_tensor(seq_mask).unsqueeze(0))  # Cb
            # x['cens']=this_centrility
            del protein['all_atom_positions']
            del protein['all_atom_mask']



            protein=tensor_to_np_dict(protein,protein.keys())
            protein['domain_name']=x['domain_name']
            protein['residue_index']=x['residue_index']
            protein['length'] = np.asarray([len(x['aatype'])])

            for k,v in protein.items():
                if protein[k].dtype==np.float32:
                    protein[k]=protein[k].astype(np.float16)

            entry=get_entry(x,protein)
            entry=cut_length(entry,256)

            train_data=train_data+entry

        output = open('/home/jorey/pdhs/dataset/S95_fbb_train_bfssecenres_fix256_'+str(i)+'.pkl', 'wb')
        print('error:' ,errortest_data)
        # 写入到文件
        pickle.dump(train_data, output)
        output.close()

        del x
        del xtensor
        del protein
        del train_data
        import gc
        gc.collect()


def test_insert():

    with open('data/cath/CATH-S40-V4.3-Split.txt', 'r') as f:
        chians=f.readlines()[0]
    chiansx=json.loads(chians)[:52000]
    train=chiansx
    from data.data_transform import np_to_tensor_dict,tensor_to_np_dict
    from model.data.data_pipeline import DataPipeline
    from data.data_transform import  atom37_to_torsion_angles,atom37_to_frames,make_atom14_positions,make_atom14_masks,get_backbone_frames

    datapipeline = DataPipeline()


    pdb_dir='//home/junyu/下载/CASP14/'
    train=glob.glob(pdb_dir+'*.pdb')
    np.set_printoptions(precision=8, suppress=True)
    error=[]


    for domain in tqdm.tqdm(train):



        pdb = domain#pdb_dir + domain+'.pdb' #
        try:
            x = datapipeline.process_pdb(pdb)
        except:
            error.append(domain)


    # gt={'train':error}
    # output ='dataset/S95_error.txt'
    # with open(output, 'w') as f:
    #     f.writelines(json.dumps(gt))






def get_protein(pdb):

    from data.data_transform import np_to_tensor_dict,tensor_to_np_dict
    from model.data.data_pipeline import DataPipeline
    from data.data_transform import  atom37_to_torsion_angles,atom37_to_frames,make_atom14_positions,make_atom14_masks,get_backbone_frames

    datapipeline = DataPipeline()


    x = datapipeline.process_pdb(pdb)
    seq_mask=np.min(np.take(x['all_atom_mask'],[0,1,2,4],axis=-1),axis=-1)
    xtensor = np_to_tensor_dict(x, ['aatype', 'all_atom_positions', 'all_atom_mask',])
    protein=atom37_to_frames(xtensor)
    protein=make_atom14_masks(protein)
    protein=make_atom14_positions(protein)
    protein=get_backbone_frames(protein)

    return protein


def write_cen_to_viotestfile(pickles='/home/jorey/pdhs/dataset/S40_vio_test_0.pkl'):
    pkl_file = open(pickles, 'rb')
    dataset = pickle.load(pkl_file)
    for i in range(len(dataset)):
        cen=centrility(torch.as_tensor(dataset[i]['atom14_gt_positions'][:, 3, :]).unsqueeze(0),
                                         mask=torch.as_tensor(dataset[i]['seqmask']).unsqueeze(0))
        dataset[i]['cen']=cen
    print('out:')
    output = open('/home/jorey/pdhs/dataset/S40_vio_test_withcen.pkl', 'wb')

    # 写入到文件
    pickle.dump(dataset, output)
    output.close()

# write_allatomspdb_files_to_pickl()
# test_insert()
if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Process PDB input and processed files output  directories')

    # Add arguments for input and output directories
    parser.add_argument('--input_dir', type=str, default='./demo/', help='Path to input PDB directory')
    parser.add_argument('--output_dir', type=str, default='./demo/', help='Path to output directory')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the function with the provided input and output directories
    write_design_pdb_files_to_pickl(args.input_dir, args.output_dir)



    #write_allatomspdb_files_to_pickl()
    # write_cen_to_viotestfile()
    # write_viotrainpdb_files_to_pickl()
    # write_fbb_pdb_files_to_pickl()


