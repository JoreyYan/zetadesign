import glob
import json
import numpy as np
import gzip
import re
import multiprocessing
import tqdm
import shutil
SENTINEL = 1
import biotite.structure as struc
import biotite.application.dssp as dssp
import biotite.structure.io.pdb.file as file
# pbar = tqdm.tqdm(total=550000)
def parse_PDB_biounits(x, sse,ssedssp,atoms=['N', 'CA', 'C'], chain=None):
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

    xyz, seq,plddts, min_resn, max_resn = {}, {}, [],  1e6, -1e6

    pdbcontents = x.split('\n')[0]
    with open(pdbcontents) as f:
        pdbcontents = f.readlines()
    for line in pdbcontents:
        #line = line.decode("utf-8", "ignore").rstrip()

        if line[:6] == "HETATM" and line[17:17 + 3] == "MSE":
            line = line.replace("HETATM", "ATOM  ")
            line = line.replace("MSE", "MET")

        if line[:4] == "ATOM":
            ch = line[21:22]
            if ch == chain or chain is None or ch==' ':
                atom = line[12:12 + 4].strip()
                resi = line[17:17 + 3]
                resn = line[22:22 + 5].strip()
                plddt=line[60:60 + 6].strip()



                x, y, z = [float(line[i:(i + 8)]) for i in [30, 38, 46]]

                if resn[-1].isalpha():
                    resa, resn = resn[-1], int(resn[:-1]) - 1 # in same pos ,use last atoms
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
    seq_, xyz_ ,sse_,ssedssp_= [], [], [], []
    dsspidx=0
    sseidx=0
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
                            xyz_.append(xyz[resn][k][atom])  #some will miss C and O ,but sse is normal,because sse just depend on CA
                        else:
                            xyz_.append(np.full(3, np.nan))
                            misschianatom=True
                    if misschianatom:
                        ssedssp_.append('-')
                        misschianatom = False
                    else:
                        try:
                            ssedssp_.append(ssedssp[dsspidx])         # if miss chain atom,xyz ,seq think is ok , but dssp miss this
                            dsspidx = dsspidx + 1
                        except:
                            print(dsspidx)


            else:
                for atom in atoms:
                    xyz_.append(np.full(3, np.nan))
                ssedssp_.append('-')


        return np.array(xyz_).reshape(-1, len(atoms), 3), N_to_AA(np.array(seq_)),np.array(sse_),np.array(ssedssp_)
    except TypeError:
        return 'no_chain', 'no_chain','no_chain'


def parse_PDB(path_to_pdb,name, input_chain_list=None):
    """
    make sure every time just input 1 line
    """
    c = 0
    pdb_dict_list = []


    if input_chain_list:
        chain_alphabet = input_chain_list
    else:
        init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                         'T',
                         'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
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
            if len(sse1)==0:
                sse1 = struc.annotate_sse(array_stack[0], chain_id='').tolist()
            ssedssp1 = dssp.DsspApp.annotate_sse(array_stack).tolist()


            xyz, seq ,sse,ssedssp= parse_PDB_biounits(biounit,sse1,ssedssp1,atoms=['N', 'CA', 'C', 'O'], chain=letter)

            # if len(sse)!=len(seq[0]):
            #     xxxx=len(seq[0])
            #     print(name)
            assert len(sse)==len(seq[0])
            assert len(ssedssp) == len(seq[0])

            if type(xyz) != str:
                concat_seq += seq[0]
                my_dict['seq_chain_' + letter] = seq[0]

                coords_dict_chain = {}
                coords_dict_chain['N'] = xyz[:, 0, :].tolist()
                coords_dict_chain['CA'] = xyz[:, 1, :].tolist()
                coords_dict_chain['C'] = xyz[:, 2, :].tolist()
                coords_dict_chain['O'] = xyz[:, 3, :].tolist()
                my_dict['coords_chain_' + letter] = coords_dict_chain

                sse=''.join(sse)
                ssedssp=''.join(ssedssp)
                my_dict['sse3' ] = sse
                my_dict['sse8'] = ssedssp
                s += 1
        #fi = biounit.rfind("/")
        my_dict['name'] = name#biounit[(fi + 1):-4]
        my_dict['num_of_chains'] = s
        my_dict['seq'] = concat_seq
        if s <= len(chain_alphabet):
            pdb_dict_list.append(my_dict)
            c += 1
    return pdb_dict_list





def parse_pdb_split_chain(pdbgzFile):

    with open(pdbgzFile) as f:


        lines = f.readlines()
        # pdbcontent = f.decode()


        pattern = re.compile('ATOM\s+\d+\s*\w+\s*[A-Z]{3,4}\s*(\w)\s*.+\n', re.MULTILINE)
        match = list(set(list(pattern.findall(lines[0]))))


    name=pdbgzFile.split('/')[-1]
        #for chain in match:
        # parse_PDB
    # match=[name[4]]
    # match=['A']
    pdb_data=parse_PDB(pdbgzFile,name,match)

    return pdb_data
def parse_pdb_split_chain_af(pdbgzFile):
    with gzip.open(pdbgzFile, 'rb') as pdbF:
        try:
            pdbcontent = pdbF.read()
        except:
            print(pdbgzFile)

        pdbcontent = pdbcontent.decode()


        pattern = re.compile('ATOM\s+\d+\s*\w+\s*[A-Z]{3,4}\s*(\w)\s*.+\n', re.MULTILINE)
        match = list(set(list(pattern.findall(pdbcontent))))


    name=pdbgzFile.split('/')[-1].split('.')[0]
        #for chain in match:
        # parse_PDB
    # match=[name[4]]
    # match=[1]
    pdb_data=parse_PDB('/media/junyu/data/perotin/aftest080_1000/'+pdbgzFile.split('/')[-1].split('.')[0]+'.pdb',name,match)

    return pdb_data

def parse_pdb_split_chain_af_3dcnn(pdbgzFile):
    with gzip.open(pdbgzFile, 'rb') as pdbF:
        try:
            pdbcontent = pdbF.read()
        except:
            print(pdbgzFile)

        pdbcontent = pdbcontent.decode()


        pattern = re.compile('ATOM\s+\d+\s*\w+\s*[A-Z]{3,4}\s*(\w)\s*.+\n', re.MULTILINE)
        match = list(set(list(pattern.findall(pdbcontent))))


    name=pdbgzFile.split('/')[-1].split('.')[0]
    namelist=[]
    for chain in match:
        namelist.append(name+'__'+chain)
    # match=[name[4]]
    # match=[1]



    return namelist
def run_net(files_path,output_path):
    """
    input is pdbgz's dir
    from pdb to jsonl
    """
    list=glob.glob(files_path+'*.pdb')#[:3110]
    data=[]
    for i in tqdm.tqdm(list):
        data_chains=parse_pdb_split_chain(i)
        #for chian in data_chains:
        data.append(data_chains[0])

    print('we want to write now')
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

    f.close()
    print('finished')

def run_netbyondif(filelist,output_path):
    with open(filelist) as f:

        lines = f.readlines()
    data=[]
    data_1=[]
    # data_2 = []
    # data_3 = []
    # data_4 = []
    # data_5 = []
    # data_6 = []
    # data_7 = []
    # data_8 = []
    # data_9 = []
    # data_10 = []
    nums_dict={1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,}

    for i in tqdm.tqdm(lines):
        data_chains,match=parse_pdb_split_chain(i.split('"')[1])

        for chian in data_chains:
            for i in match:
                meanplddt = round(float(np.asarray(chian['plddts_chain_' + i]).mean()),2)
                data.append({'name':chian['name'],'lens':len(chian['seq']),'meanplddt':meanplddt})
                if int(meanplddt/10)==1:
                    #data_1.append(chian)
                    nums_dict[1]=nums_dict[1]+1
                elif int(meanplddt/10)==2:
                    #data_2.append(chian)
                    nums_dict[2] = nums_dict[2] + 1
                elif int(meanplddt / 10) == 3:
                    #data_3.append(chian)
                    nums_dict[3] = nums_dict[3] + 1
                elif int(meanplddt / 10) == 4:
                    #data_4.append(chian)
                    nums_dict[4] = nums_dict[4] + 1
                elif int(meanplddt / 10) == 5:
                    #data_5.append(chian)
                    nums_dict[5] = nums_dict[5] + 1
                elif int(meanplddt / 10) == 6:
                    #data_6.append(chian)
                    nums_dict[6] = nums_dict[6] + 1
                elif int(meanplddt / 10) == 7:
                    #data_7.append(chian)
                    nums_dict[7] = nums_dict[7] + 1
                elif int(meanplddt / 10) == 8:
                    #data_8.append(chian)
                    nums_dict[8] = nums_dict[8] + 1
                elif int(meanplddt / 10) == 9:
                    #data_9.append(chian)
                    nums_dict[9] = nums_dict[9] + 1
                elif int(meanplddt / 10) == 10:
                    #data_10.append(chian)
                    nums_dict[10] = nums_dict[10] + 1
                else:
                    print(chian['name'])


    #         data.append(chian)
    #
    f.close()
    output_pathindex=output_path+filelist.split('/')[-1].split('.')[0]+'_detail.jsonl'
    print('we want to write now')
    with open(output_pathindex, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

    f.close()
    print(nums_dict)
    # count(output_pathindex)
    print('finished')
def list_of_groups(list_info, per_list_len):
    '''
    :param list_info:   列表
    :param per_list_len:  每个小列表的长度
    :return:
    '''
    list_of_group = zip(*(iter(list_info),) *per_list_len)
    end_list = [list(i) for i in list_of_group] # i is a tuple
    count = len(list_info) % per_list_len
    end_list.append(list_info[-count:]) if count !=0 else end_list
    return end_list

def count(filelist):
    with open(filelist) as f:

        lines = f.readlines()
    plddts=[]

    for i in tqdm.tqdm(lines):
        pl=json.loads(i)['meanplddt']
        plddts.append(int(pl/10))

    for i in range(10):
        print('counts '+str(i),plddts.count(i))

def run_net_aftest(files_path,output_path):
    """
    input is pdbgz's dir
    """
    with open(files_path) as f:
        lines = f.readlines()
    data=[]
    for i in tqdm.tqdm(lines):

        data_chains=parse_pdb_split_chain_af('/media/junyu/data/point_cloud/'+i.split('"')[1])
        for chian in data_chains:
            data.append(chian)

    # print('we want to write now')
    # with open(output_path, 'w') as f:
    #     for entry in data:
    #         f.write(json.dumps(entry) + '\n')
    #
    # f.close()
    # print('finished')

    output_pathindex = output_path + str(80) + 'bigthanclass_1000.text'
    print('we want to write now')
    with open(output_pathindex, 'w') as f:
        for entry in data:
            f.write(entry + '\n')


    f.close()

if __name__ == "__main__":
    files_path='/media/junyu/data/perotin/chain_set/AFDATA/details/80bigthanclass_1000.jsonl' #'/home/junyu/下载/splits/'#
    output_path='/media/junyu/data/perotin/chain_set/'



    # run_net_aftest(files_path,output_path)

    fakedata='//home/oem/pdb-tools/pdbtools/fixed/'
    run_net(fakedata,output_path+'tim184.jsonl')



        #
        # f.close()
        # # print(nums_dict)
        # print('finished ' +str(i))


    # alllist=list_of_groups(lists,10000)

    # for i in range(len(alllist)):
    #     thislist=alllist[i]
    #     with open(output_path+'_'+str(i)+'.jsonl', 'w') as f:
    #         for entry in thislist:
    #             f.write(json.dumps(entry) + '\n')
    #
    #     f.close()
    #     # print(nums_dict)
    #     print('finished ' +str(i))

    # _processes = []



    # q = multiprocessing.Queue()
    #
    # proc.start()
    # for eachlist in alllist:
    #     _process = multiprocessing.Process(target=run_netbyondif, args=(eachlist,))
    #     _process.start()


    # run_netbyondif(lists,output_path)