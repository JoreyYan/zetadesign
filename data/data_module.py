import logging
import time

import torch

import model.np.residue_constants as rc
import pickle
import numpy as np
from model.Rigid import Rigid,Rotation
import random
from data.data_transform import np_to_tensor_dict


class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, pickles):
        pkl_file = open(pickles, 'rb')
        dataset = pickle.load(pkl_file)

        self.data = dataset
        random.shuffle(self.data)
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]

        return data
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)


class StructureDataset():
    def __init__(self, pickles, verbose=True, truncate=None, max_length=100,min_length=25,
                 part=None,**kwargs):
        alphabet_set = rc.restypes_with_x
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }
        self.data=[]
        logging.info('now loading ')  #,pickles
        logging.info(pickles)  # ,pickles
        pkl_file = open(pickles, 'rb')
        dataset = pickle.load(pkl_file)
        random.shuffle(dataset)
        start = time.time()
        for i, line in enumerate(dataset):
                entry = line

                #name = entry['domain_name']
                #seq = entry['seq'][0].decode('UTF-8')

                # Check if in alphabet
                #bad_chars = set([s for s in seq]).difference(alphabet_set)
                bad_chars=[]
                length=int(entry['length'])
                # length=int(entry[1]['aatype'].shape[0])
                if len(bad_chars) == 0:

                    if length <= max_length and length >= min_length:
                        if True:
                            self.data.append(entry)
                        else:
                            discard_count['bad_seq_length'] += 1
                    else:
                        discard_count['too_long'] += 1
                else:
                    #print(name, bad_chars, entry[seq_name])
                    discard_count['bad_chars'] += 1

                # Truncate early
                if truncate is not None and len(self.data) == truncate:
                    return

                if verbose and (i + 1) % 1000 == 0:
                    elapsed = time.time() - start
                    print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i + 1, elapsed))

        print('discarded', discard_count)

        if part is not None:
            random.shuffle(self.data)
            self.data=self.data[part:]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class StructureLoader():
    def __init__(self, dataset, batch_size=100, shuffle=True,**kwargs
                 ):
        self.shuffle=shuffle

        self.dataset = dataset
        self.size = len(dataset)
        self.lengths=[]
        for i in range(self.size):
            #thislength=dataset[i][1]['aatype'].shape[0]
            thislength=dataset[i]['length']
            self.lengths.append(thislength)


        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []

        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)

            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)

        self.clusters=clusters
        if self.shuffle:
            random.shuffle(self.clusters)

    def _shuffle(self):
        random.shuffle(self.clusters)

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
        # for i in range(len(self.lengths)):
        #     batch=self.dataset[i]
            yield batch


class StructureDataset_allatoms():
    def __init__(self, pickles, verbose=True, truncate=None, max_length=100,min_length=25,
                 part=None,**kwargs):
        alphabet_set = rc.restypes_with_x
        discard_count = {
            'bad_chars': 0,
            'bad_length': 0,
            'bad_seq_length': 0
        }
        self.data=[]
        logging.info('now loading '+pickles)
        pkl_file = open(pickles, 'rb')
        dataset = pickle.load(pkl_file)
        start = time.time()
        for i, line in enumerate(dataset):
                entry = line
                name = entry['domain_name']
                length=entry['aatype'].shape[0]
                if length <= max_length and length >= min_length :
                    self.data.append(entry)
                else:
                    discard_count['bad_length'] += 1

                if verbose and (i + 1) % 1000 == 0:
                    elapsed = time.time() - start
                    print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i + 1, elapsed))
        random.shuffle(self.data)
        print('discarded', discard_count)
        if part is not None:
            self.data=self.data[324:]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



def allatoms_tied_features(batch,device,require_angles=True):
    '''

    :param batch: output of dataloader
    :param require_angles: True in joint Model
    :return:
    '''
    pad0_list = ['rigidgroups_gt_frames', 'rigidgroups_gt_exists', 'atom14_atom_exists', 'atom14_gt_exists',
                 'atom14_gt_positions', 'atom14_alt_gt_positions', 'atom14_alt_gt_exists', 'atom14_atom_is_ambiguous',
                 'backbone_rigid_tensor', 'backbone_rigid_mask', 'torsion_angles_sin_cos', 'alt_torsion_angles_sin_cos',
                 'torsion_angles_mask', 'residx_atom14_to_atom37', 'rigidgroups_alt_gt_frames']
    B=len(batch)
    if B>=1:
        L_max=int(np.max([i['aatype'].shape[0] for i in batch]))

        # get backbone atoms
        x_pad = [np.pad(i['atom14_gt_positions'][:,:4,:],[[0,L_max-i['aatype'].shape[0]], [0,0], [0,0]], 'constant', constant_values=-1000.) for i in  batch]
        x = np.concatenate(np.expand_dims(x_pad,0), 0)  # [B, L, 4, 3]
        x = torch.tensor(x, device=device,dtype=torch.float32)


        # get aatype
        aatypes = [np.pad(i['aatype'], [0, L_max - i['aatype'].shape[0]], 'constant',constant_values=20) for i in batch]
        aatypes = np.concatenate(np.expand_dims(aatypes,0), 0)  # [B, L, ]
        aatypes = torch.tensor(aatypes, device=device)



        # get seq_mask
        seq_masks = [np.pad(np.min(np.take(i['atom14_atom_exists'],[0,1,2,4],axis=-1),axis=-1), [0, L_max - i['aatype'].shape[0]], 'constant',constant_values=0) for i in batch]
        seq_masks = np.concatenate(np.expand_dims(seq_masks,0), 0)  # [B, L, ]
        seq_masks = torch.tensor(seq_masks, device=device)

        # get residx
        residue_indexs = [np.pad(i['residue_index'], [0, L_max - i['aatype'].shape[0]], 'constant',constant_values=-100) for i in batch]
        residue_indexs = np.concatenate(np.expand_dims(residue_indexs,0), 0)  # [B, L, ]
        residue_indexs = torch.tensor(residue_indexs, device=device)



        gt={}
        for feat in pad0_list:

            if len(batch[0][feat].shape)==1:
                feat_pad = [np.pad(i[feat], [0, L_max - i['aatype'].shape[0]], 'constant',constant_values=0) for i in batch]
            elif len(batch[0][feat].shape)==2:
                feat_pad = [np.pad(i[feat], [[0, L_max - i['aatype'].shape[0]], [0,0]], 'constant', constant_values=0) for i in batch]
            elif len(batch[0][feat].shape)==3:
                feat_pad = [np.pad(i[feat], [[0, L_max - i['aatype'].shape[0]], [0,0], [0,0]], 'constant', constant_values=0) for i in batch]
            elif len(batch[0][feat].shape)==4:
                feat_pad = [np.pad(i[feat], [[0, L_max - i['aatype'].shape[0]], [0,0], [0,0], [0,0]], 'constant', constant_values=0) for i in batch]
            feat_pad = np.concatenate(np.expand_dims(feat_pad,0), 0)  # [B, L, ]
            feat_pad = torch.tensor(feat_pad, device=device)

            gt[feat]=feat_pad
            gt['residue_index']=residue_indexs.type(torch.long)
            gt["aatype"]=aatypes

    # else:
    #     x=torch.tensor(batch[0]['atom14_gt_positions'][:,:4,:],device=device).type(torch.float32).unsqueeze(0)
    #     aatypes=torch.tensor(batch[0]['aatype'],device=device).unsqueeze(0)
    #     seq_masks=torch.tensor(np.min(np.take(batch[0]['atom14_atom_exists'],[0,1,2,4],axis=-1),axis=-1),device=device).unsqueeze(0)
    #     residue_indexs=torch.tensor(batch[0]['residue_index'],device=device).unsqueeze(0)
    #
    #     del batch[0]['domain_name']
    #
    #     gtall=batch[0]
    #     gtall=np_to_tensor_dict(gtall,gtall.keys())
    #     gt={}
    #     for feat in pad0_list:
    #         gt[feat] =gtall[feat].to(device)
    #         gt['residue_index']=residue_indexs.squeeze(0).type(torch.long).to(device)
    #         gt["aatype"]=aatypes.squeeze(0).to(device)



    return x, aatypes, seq_masks, residue_indexs,gt


def tied_features_J(batch,device,require_angles=True):
    '''

    :param batch: output of dataloader
    :param require_angles: True in joint Model
    :return:
    '''
    B=len(batch)
    L_max=int(np.max([i['length'] for i in batch]))

    # get backbone atoms
    x_pad = [np.pad(i['backbone_atom_positions'],[[0,L_max-i['length']], [0,0], [0,0]], 'constant', constant_values=-10000.) for i in  batch]
    x = np.concatenate(np.expand_dims(x_pad,0), 0)  # [B, L, 4, 3]
    x = torch.tensor(x, device=device)

    # get aatype
    aatypes = [np.pad(i['aatype'], [0, L_max - i['length']], 'constant',constant_values=20) for i in batch]
    aatypes = np.concatenate(np.expand_dims(aatypes,0), 0)  # [B, L, ]
    aatypes = torch.tensor(aatypes, device=device)

    # get seq_mask
    seq_masks = [np.pad(i['seqmask'], [0, L_max - i['length']], 'constant',constant_values=0) for i in batch]
    seq_masks = np.concatenate(np.expand_dims(seq_masks,0), 0)  # [B, L, ]
    seq_masks = torch.tensor(seq_masks, device=device)

    # get residx
    residue_indexs = [np.pad(i['residue_index'], [0, L_max - i['length']], 'constant',constant_values=-100) for i in batch]
    residue_indexs = np.concatenate(np.expand_dims(residue_indexs,0), 0)  # [B, L, ]
    residue_indexs = torch.tensor(residue_indexs, device=device)

    if not require_angles:
        return x,aatypes,seq_masks,residue_indexs
    else:
        # get angles
        torsion_angles = [np.pad(i['torsion_angles_sin_cos'], [[0, L_max - i['length']],[0,0],[0,0]], 'constant', constant_values=0.) for i in
                         batch]
        torsion_angles = np.concatenate(np.expand_dims(torsion_angles, 0), 0)  # [B, L, 7,2,]
        torsion_angles=torch.tensor(torsion_angles,device=device)

        # get alt angles
        alt_torsion_angles = [np.pad(i['alt_torsion_angles_sin_cos'], [[0, L_max - i['length']],[0,0],[0,0]], 'constant', constant_values=0.) for i in
                         batch]
        alt_torsion_angles = np.concatenate(np.expand_dims(alt_torsion_angles, 0), 0)  # [B, L, 7,2,]
        alt_torsion_angles = torch.tensor(alt_torsion_angles, device=device)

        # get alt angles
        torsion_angle_masks = [
            np.pad(i['torsion_angles_mask'], [[0, L_max - i['length']], [0, 0], ], 'constant',
                   constant_values= 0.) for i in
            batch]
        torsion_angle_masks = np.concatenate(np.expand_dims(torsion_angle_masks, 0), 0)  # [B, L, 7,]

        torsion_angle_masks = torch.tensor(torsion_angle_masks, device=device)

        return x, aatypes, seq_masks, residue_indexs,torsion_angles,alt_torsion_angles,torsion_angle_masks
def tied_features(batch,device,require_angles=True):
    '''

    :param batch: output of dataloader
    :param require_angles: True in joint Model
    :return:
    '''
    B=len(batch)
    try:
        L_max=int(np.max([i['length'] for i in batch]))
    except:
        print(L_max)


    # get backbone atoms
    x_pad = [np.pad(i['backbone_atom_positions'],[[0,L_max-i['length']], [0,0], [0,0]], 'constant', constant_values=-10000.) for i in  batch]
    x = np.concatenate(np.expand_dims(x_pad,0), 0)  # [B, L, 4, 3]
    x = torch.tensor(x, device=device,dtype=torch.float)

    # get aatype
    aatypes = [np.pad(i['aatype'], [0, L_max - i['length']], 'constant',constant_values=20) for i in batch]
    aatypes = np.concatenate(np.expand_dims(aatypes,0), 0)  # [B, L, ]
    aatypes = torch.tensor(aatypes, device=device,dtype=torch.long)

    # get seq_mask
    seq_masks = [np.pad(i['seqmask'], [0, L_max - i['length']], 'constant',constant_values=0) for i in batch]
    seq_masks = np.concatenate(np.expand_dims(seq_masks,0), 0)  # [B, L, ]
    seq_masks = torch.tensor(seq_masks, device=device,dtype=torch.float)

    # get residx
    residue_indexs = [np.pad(i['residue_index'], [0, L_max - i['length']], 'constant',constant_values=-100) for i in batch]
    residue_indexs = np.concatenate(np.expand_dims(residue_indexs,0), 0)  # [B, L, ]
    residue_indexs = torch.tensor(residue_indexs, device=device,dtype=torch.float)


    if not require_angles:
        return x,aatypes,seq_masks,residue_indexs
    else:



        # get angles
        torsion_angles = [np.pad(i['torsion_angles_sin_cos'], [[0, L_max - i['length']],[0,0],[0,0]], 'constant', constant_values=0.) for i in
                         batch]
        torsion_angles = np.concatenate(np.expand_dims(torsion_angles, 0), 0)  # [B, L, 7,2,]
        torsion_angles=torch.tensor(torsion_angles,device=device,dtype=torch.float32)

        # get alt angles
        alt_torsion_angles = [np.pad(i['alt_torsion_angles_sin_cos'], [[0, L_max - i['length']],[0,0],[0,0]], 'constant', constant_values=0.) for i in
                         batch]
        alt_torsion_angles = np.concatenate(np.expand_dims(alt_torsion_angles, 0), 0)  # [B, L, 7,2,]
        alt_torsion_angles = torch.tensor(alt_torsion_angles, device=device,dtype=torch.float)

        # get alt angles
        torsion_angle_masks = [
            np.pad(i['torsion_angles_mask'], [[0, L_max - i['length']], [0, 0], ], 'constant',
                   constant_values= 0.) for i in
            batch]
        torsion_angle_masks = np.concatenate(np.expand_dims(torsion_angle_masks, 0), 0)  # [B, L, 7,]

        torsion_angle_masks = torch.tensor(torsion_angle_masks, device=device,dtype=torch.float32)

        #residx_atom14_to_atom37 is used to restore full atoms
        pad0_list = [ 'atom14_atom_exists', 'atom14_gt_exists',
                     'atom14_gt_positions', 'atom14_alt_gt_positions', 'atom14_alt_gt_exists',
                     'atom14_atom_is_ambiguous','rigidgroups_alt_gt_frames','rigidgroups_gt_frames','rigidgroups_gt_exists',
                      'b_factors','sse8','sse3','cens',
                      'residx_atom14_to_atom37'
                     ]
        gt={}



        for feat in pad0_list:




            if len(batch[0][feat].shape)==1:
                feat_pad = [np.pad(i[feat], [0, L_max - i['aatype'].shape[0]], 'constant',constant_values=0) for i in batch]
            elif len(batch[0][feat].shape)==2:
                feat_pad = [np.pad(i[feat], [[0, L_max - i['aatype'].shape[0]], [0,0]], 'constant', constant_values=0) for i in batch]
            elif len(batch[0][feat].shape)==3:
                feat_pad = [np.pad(i[feat], [[0, L_max - i['aatype'].shape[0]], [0,0], [0,0]], 'constant', constant_values=0) for i in batch]
            elif len(batch[0][feat].shape)==4:
                feat_pad = [np.pad(i[feat], [[0, L_max - i['aatype'].shape[0]], [0,0], [0,0], [0,0]], 'constant', constant_values=0) for i in batch]
            feat_pad = np.concatenate(np.expand_dims(feat_pad,0), 0)  # [B, L, ]
            feat_pad = torch.tensor(feat_pad, device=device)

            gt[feat]=feat_pad




        batch = {
            # 'gtframes': rigidgroups_gt_frames,
            # 'gtframes_mask': rigidgroups_gt_exists,
            'aatype': aatypes,
            'gt_angles_sin_cos': torsion_angles,
            'gt_angles_mask': torsion_angle_masks,
            'residue_index':residue_indexs.type(torch.long)
        }
        batch.update(gt)


        return x, aatypes, seq_masks, residue_indexs,batch

def gt_batch(x, aatypes, seq_masks, torsion_angles,torsion_angle_masks,eps,device):


    gt_frame=Rigid.from_3_points(p_neg_x_axis=x[..., 0, :],origin=x[..., 1, :],p_xy_plane=x[..., 2, :],eps=eps,)
    gt_frame_mask=seq_masks.to(device)
    gt_aatype=aatypes.to(device)
    gt_angles=torsion_angles.to(device)
    gt_angles_mask=torsion_angle_masks.to(device)

    batch={
        'gtframes':gt_frame,
        'gtframes_mask':gt_frame_mask,
        'gt_aatype':gt_aatype,
        'gt_angles_sin_cos':gt_angles,
        'gt_angles_mask':gt_angles_mask,
    }


    return batch




