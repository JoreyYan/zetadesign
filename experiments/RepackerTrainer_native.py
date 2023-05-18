import glob
import json
import random

import numpy as np
import torch
import torch.nn as nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from torch.utils.data import DataLoader

import ml_collections as mlc
from data import builder_for_AL
import tqdm
from model.utils import AverageMeter
import pickle

import math


from model.network.Repacker import Repacker,Repacker_iter
from data.data_module import StructureDataset,StructureLoader,tied_features,gt_batch,GetLoader
import logging
from set_logger import set_logger1
set_logger1(log_path='../logs/mlvio_huber_rmsd_degreesz_finetune_native_testonly_E_norm.log')
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

from model.np.residue_constants import restypes_with_x,restypes,restype_order


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 注意如果 device = torch.device("cuda")，则环境变量CUDA_VISIBLE_DEVICES中指定的全部GPU都会被拿来使用。
# 也可以通过 "cuda:0" 、"cuda:1"等指定环境变量CUDA_VISIBLE_DEVICES中指定的多块GPU中的某一块。

class argsx():
    def __init__(self):


        self.just_valastrain=True  # for debug using little data


        self.device=device
        print(device)

        # trainmodel
        self.trainmodel='R'
        datapath=  '/home/jorey/pdhs/dataset/'     #'//home/omnisky/data/everyOne/yjy/pdhs/dataset/'  #           #
        pro_path=  '/home/jorey/pdhs/'   #  '/home/omnisky/data/everyOne/yjy/pdhs/' #


        #data
        self.dataset={
            'trainset':datapath+'S40_trainfinetunetest_bfssecen_fix256.pkl',#S40_train_bfssecen_fix256_,

            'testset':datapath+'/S40_train_bfssecen_fix256casp14.pkl',  #dataset_test_0 jusr is casp14
            'designset': pro_path+'/data/CASP_Result/CASP14_Native.pkl',
            'max_length':257,
            'batch_size':513,
            'shuffle':True
        }

        #output
        self.output_dir='//home/junyu/casp/mpnn_repacker/'

        eps=1e-8
        self.loss={
            'aux_f':0.1,
            'fape_f':1,

            'angle_f':1,
            'chi_weight' : 1,
            'bb_torsion_weight' : 1,
            'angle_norm_weight' : 0.1,
            'aatempweight':0.1,
            'kind':len(restypes_with_x)
        }
        self.R_param={
            'n_module_str':6,
            'msa_layer':2,
            'IPA_layer':2,
            'd_msa':128,
            'd_ipa':128,
            'n_head_msa':8,
            'n_head_ipa':8,
            'r_ff':2,
            'p_drop':0.1,

            "r_epsilon": 1e-8,  # 1e-12,
            #angles_module
            'c_s': 128,
            "c_resnet": 128,
            "no_resnet_blocks": 4,
            "no_angles": 7,
            "trans_scale_factor": 10.,
            'bf_scale_factor':10.,
            "a_epsilon": 1e-12,  # 1e-12,
            'bbrelaxdistance':0,


            'device':self.device,
            'pairwise_repr_dim': 32,
            'require_pairwise_repr': False,

            'loss_factor':self.loss

        }

        self.str_encoder_param={
            'node_features' : 256,
            'edge_features': 256,
            'hidden_dim':256,
            'num_encoder_layers':3,
            'augment_eps':0.,
            'k_neighbors' :48,
            'dropout' : 0.1,
            'vocab' : len(restypes_with_x),
            'use_ESSE' : False,
            'use_Eaatype': True,
            'use_tri':True,
            'use_rbf':True,
            'use_sse':False,
        }



        self.R_scheduler= {'scheduler':{
                'type': 'CosLR',
                'kwargs': {
                    'epochs': 30,
                    'initial_epochs': 0,
                    'warming_up_init_lr': 0.00001
                }},
                'start_epoch':0}

        self.R_Optimzer={'optimizer':{
                'type': 'Adam',
                'kwargs': {
                    'lr': 0.0001,
                    'weight_decay': 0.0002
                }}}

        self.vis= {
            'logging_per_update':10
        }

        self.checkpoint= {
            'save_path':pro_path+'/save/R/',
            'R_per_weight': pro_path+'/save/R/R_withbf_bfssecen_huberrmsd_lit_ml_finetune0_9.pth',
            'resume_G':False,



        }
        # R_Repacker_128_6_G256_fape_error_1__1_3   S40 Train without vio, and with fape




        self.loss_config=mlc.ConfigDict(
            {
                "loss": {
                    "fape": {
                        "backbone": {
                            "clamp_distance": 10.0,
                            "loss_unit_distance": 10.0,
                            "weight": 0.5,
                        },
                        "sidechain": {
                            "clamp_distance": 10.0,
                            "length_scale": 10.0,
                            "weight": 0.5,
                        },
                        "eps": 1e-4,
                        "weight": 1.0,
                    },
                        }
            })





    def load_module(self,module,module_param,Optimzer_param,scheduler_param,Resume_pth=None):
        # build G model
        model = module(**module_param,Str_encoder_param=self.str_encoder_param)
        self.model = model.to(device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model =  nn.DataParallel(model) # device_ids will include all GPU devices by default
        self.model = self.model.to(device)


        self.optimizer, self.scheduler = builder_for_AL.build_opti_sche(model, **Optimzer_param, **scheduler_param)

        # load checkpoints
        if Resume_pth is not None:
            checkpoint_path=Resume_pth
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            checkpoint_model = checkpoint["base_model"]
            model_dict = model.state_dict()
            pretrained_dict = {k[7:]: v for k, v in checkpoint_model.items() if k[7:] in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            # self.optimizer.load_state_dict(checkpoint["optimizer"])

            del model_dict
            logging.info("load model checkpoints")
        else:
            logging.info("not load model checkpoints")

        # beginning to train
        #load G
        if self.checkpoint['resume_G']:
            Gcheckpoint = torch.load(self.checkpoint['G_per_weight'], map_location=self.device)
            checkpoint_model = Gcheckpoint["model_state_dict"]
            model_dict = model.state_dict()
            pretrained_dict = {'G.' + k: v for k, v in checkpoint_model.items() if 'G.' + k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)






    def load_dataset(self):

        if self.just_valastrain:
            testloader = StructureLoader(StructureDataset(self.dataset['testset'],**self.dataset),**self.dataset)

            self.trainloader=testloader
            self.testloader=testloader


        else:
            trainloader = StructureLoader(StructureDataset(self.dataset['trainset'], **self.dataset)  ,**self.dataset)
            testloader =  StructureLoader(StructureDataset(self.dataset['testset']  , **self.dataset) ,**self.dataset)
            self.trainloader = trainloader
            self.testloader=testloader

    def load_dataset_mulit(self, i):

        nums = len(self.dataset['trainset'])
        dataset = self.dataset['trainset'][i]
        print('left trainset:', self.dataset['trainset'][i:])

        if self.just_valastrain:
            testloader = StructureLoader(StructureDataset(self.dataset['testset'], **self.dataset), **self.dataset)

            self.trainloader = testloader
            self.testloader = testloader


        else:
            # # trainloader = StructureLoader(StructureDataset(dataset, **self.dataset), **self.dataset)
            # torch_data=GetLoader(dataset)
            #trainloader = DataLoader(torch_data, batch_size=1, shuffle=True, drop_last=True, num_workers=2)

            trainloader = StructureLoader(StructureDataset(dataset, **self.dataset),
                                         **self.dataset)
            testloader = StructureLoader(StructureDataset(self.dataset['testset'], **self.dataset),
                                         **self.dataset)

            # torch_data=GetLoader(self.dataset['testset'])
            # testloader = DataLoader(torch_data, batch_size=1, shuffle=True, drop_last=True, num_workers=6)

            self.trainloader = trainloader
            self.testloader = testloader

    def train_step(self,epoch):
        with torch.no_grad():
            # self.load_dataset_mulit(i)
            # tools
            acc = AverageMeter(
                [ 'Ca_lddt', 'angles_acc1', 'angles_acc2', 'angles_acc3', 'angles_acc4',
                 'angles_acc5', 'angles_acc6', 'angles_acc7'])
            losses = AverageMeter(['q loss','ca loss','chi loss','bb angles loss','fames loss','side fape loss'])
            logging.info(60*'-' )
            logging.info(f"\n[Train] Start training epoch {epoch}", )
            logging.info(60*'-' )
            self.model.train()
            t = tqdm.tqdm(self.trainloader)


            self.scheduler.step(epoch)

            #train
            for proteins in t:
                self.optimizer.zero_grad()






                x, aatypes, seq_masks, residue_indexs,gt_batchs=tied_features(
                    proteins,self.device,False if self.trainmodel=='G' else True)


                inputs = {'X': x,'S': aatypes,'mask': seq_masks,'residue_idx': residue_indexs}  #'SSE8_seq':gt_batchs['sse8'],'SSE3_seq':gt_batchs['sse3'],


                loss,result=self.model(inputs,gt_batchs)
                loss=torch.mean(loss)
                # loss.backward()
                # self.optimizer.step()

                # acc.update([ result['ca_lddt'].item(),result['Angle_accs'][0].item(), result['Angle_accs'][1].item(), result['Angle_accs'][2].item()
                #                , result['Angle_accs'][3].item(), result['Angle_accs'][4].item(),result['Angle_accs'][5].item(),
                #             result['Angle_accs'][6].item()])


                losses.update([result['bfloss'].mean(),result['ca_loss'].mean(),result['chi_loss'].mean(),result['bb_angle_loss'].mean(),
                               result['violations'].mean(),result['fapeloss'].mean()])

                if self.global_step%self.vis['logging_per_update']==0:

                    logging.info(
                        "bfloss:%.2f;ca_loss:%.2f;chi_angle_loss:%.2f;bb_angle_loss:%.2f;violations:%.5f;fape:%.3f;step %d；lr:%4f" % (
                            float(losses.avg(0)),  float(losses.avg(1)), float(losses.avg(2)), float(losses.avg(3)),float(losses.avg(4)),float(losses.avg(5)),
                             self.global_step,
                            self.optimizer.param_groups[0]['lr']))

                    # logging.info(
                    #     "ca_lddt:%.2f;Angle_accs1:%.2f;2:%.2f;3:%.2f;chi1:%.2f;chi2:%.2f;chi3:%.2f;chi4:%.2f;" % (
                    #         float(acc.avg(0)), float(acc.avg(1)), float(acc.avg(2)), float(acc.avg(3)), float(acc.avg(4))
                    #     , float(acc.avg(5)), float(acc.avg(6)), float(acc.avg(7)))
                    #     )

                self.global_step=self.global_step+1

    def val_step(self, epoch):
        with torch.no_grad():
            # tools
            acc = AverageMeter(
                ['Ca_lddt', 'angles_acc1', 'angles_acc2', 'angles_acc3', 'angles_acc4',
                 'angles_acc5', 'angles_acc6', 'angles_acc7'])
            losses = AverageMeter(['q loss', 'ca loss', 'chi loss', 'bb angles loss', 'violations', 'fape'])
            logging.info(f"\n[VAL] Start val epoch {epoch}", )
            self.model.eval()
            t = tqdm.tqdm(self.testloader)
            val_step=0
            # test
            for proteins in t:

                self.optimizer.zero_grad()

                # print(proteins['domain_name'])
               # proteins=[proteins]
                x, aatypes, seq_masks, residue_indexs, gt_batchs = tied_features(
                    proteins, self.device, False if self.trainmodel == 'G' else True)


                input = {'X': x, 'S': aatypes, 'mask': seq_masks, 'residue_idx': residue_indexs}  #'SSE8_seq':gt_batchs['sse8'],'SSE3_seq':gt_batchs['sse3'],

                loss, result = self.model(input, gt_batchs,)

                # acc.update([result['ca_lddt'], result['Angle_accs'][0], result['Angle_accs'][1],
                #             result['Angle_accs'][2]
                #                , result['Angle_accs'][3], result['Angle_accs'][4],
                #             result['Angle_accs'][5],
                #             result['Angle_accs'][6]])
                losses.update([result['bfloss'].mean(), result['ca_loss'].mean(), result['chi_loss'].mean(),
                               result['bb_angle_loss'].mean(),result['violations'].mean(),result['fapeloss'].mean()])

                if val_step % self.vis['logging_per_update'] == 0:
                    logging.info(
                        "quanteron_loss:%.2f;ca_loss:%.2f;chi_angle_loss:%.2f;bb_angle_loss:%.2f;violations:%.5f;fapeloss:%.3f;step %d；lr:%4f" % (
                            float(losses.avg(0)), float(losses.avg(1)), float(losses.avg(2)), float(losses.avg(3)),float(losses.avg(4)),float(losses.avg(5)),
                            self.global_step,
                            self.optimizer.param_groups[0]['lr']))

                    # logging.info(
                    #     "ca_lddt:%.2f;Angle_accs1:%.2f;2:%.2f;3:%.2f;chi1:%.2f;chi2:%.2f;chi3:%.2f;chi4:%.2f;" % (
                    #         float(acc.avg(0)), float(acc.avg(1)), float(acc.avg(2)), float(acc.avg(3)),
                    #         float(acc.avg(4)), float(acc.avg(5)), float(acc.avg(6)), float(acc.avg(7)))
                    # )
                val_step=val_step+1
            logging.info(
                "bfloss:%.2f;ca_loss:%.2f;chi_angle_loss:%.2f;bb_angle_loss:%.2f;violations:%.5f;fapeloss:%.3f;step %d；lr:%4f" % (
                    float(losses.avg(0)), float(losses.avg(1)), float(losses.avg(2)), float(losses.avg(3)),
                    float(losses.avg(4)), float(losses.avg(5)),
                    self.global_step,
                    self.optimizer.param_groups[0]['lr']))

            # logging.info(
            #     "ca_lddt:%.2f;Angle_accs1:%.2f;2:%.2f;3:%.2f;chi1:%.2f;chi2:%.2f;chi3:%.2f;chi4:%.2f;" % (
            #         float(acc.avg(0)), float(acc.avg(1)), float(acc.avg(2)), float(acc.avg(3)),
            #         float(acc.avg(4)), float(acc.avg(5)), float(acc.avg(6)), float(acc.avg(7)))
            # )

    def save_checkpoints(self,epoch):
        prefix=self.trainmodel+str('_withbf_bfssecen_huberrmsd_lit_ml_finetune_native_E')+str(self.R_param['bbrelaxdistance'])+'_'+str(epoch)
        torch.save({
            'base_model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),

            'epoch': epoch,
            # 'metrics' : val_metrics.state_dict() if val_metrics is not None else dict(),

        }, os.path.join(self.checkpoint['save_path'], prefix + '.pth'))


    def show_lr(self):

        SEQGenerator, optimizer_G, scheduler_G = self.load_G()

        self.optimizer_G = optimizer_G
        self.scheduler_G = scheduler_G



        # build dataset
        self.load_dataset()

        self.lr=[]

        self.global_step = 0
        SEQGenerator.train()
        for epoch in range(self.G_scheduler['start_epoch'], self.G_scheduler['scheduler']['kwargs']['epochs'] + 1):

            logging.info(f"\n just test lr epoch {epoch}", )

            t = tqdm.tqdm(self.trainloader)

            if self.G_scheduler['scheduler']['type'] != 'function':
                if isinstance(self.scheduler_G, list):
                    for item in self.scheduler_G:
                        item.step(epoch)
                else:
                    self.scheduler_G.step(epoch)

            #train
            for proteins in t:
                lr=self.optimizer_G.param_groups[0]['lr']
                self.lr.append(lr)

                if self.global_step % self.vis['logging_per_update'] == 0:
                    logging.info(
                        "step %d；lr:%4f" % (self.global_step,
                            self.optimizer_G.param_groups[0]['lr']))
                self.global_step=self.global_step+1

    def train_R(self,  ):

        #load MODEL
        logging.info(self.dataset)
        logging.info(self.R_Optimzer)
        logging.info(self.str_encoder_param)
        logging.info(self.loss)
        logging.info(self.checkpoint)
        self.load_module(Repacker,self.R_param,self.R_Optimzer,self.R_scheduler,self.checkpoint['R_per_weight']) ##self.checkpoint['R_per_weight']

        # build dataset
        self.load_dataset()
        self.global_step=self.R_scheduler['start_epoch']*len(self.trainloader)
        self.model.zero_grad()

        for epoch in range(self.R_scheduler['start_epoch'], self.R_scheduler['scheduler']['kwargs']['epochs'] ):
            # build dataset
            self.load_dataset()
            # random.shuffle(self.dataset['trainset'])
            # for i in range(len(self.dataset['trainset'])):

            self.train_step(epoch)
            self.save_checkpoints(epoch)
            self.val_step(epoch)




    def design_step(self, epoch,pre_name=None
                    ,pre_aatype=None):

        with torch.no_grad():
            # tools
            acc = AverageMeter(
                ['Ca_lddt', 'angles_acc1', 'angles_acc2', 'angles_acc3', 'angles_acc4',
                 'angles_acc5', 'angles_acc6', 'angles_acc7'])
            losses = AverageMeter(['q loss', 'ca loss', 'chi loss', 'bb angles loss'])
            logging.info(f"\n[VAL] Start val epoch {epoch}", )
            self.model.eval()
            t = tqdm.tqdm(self.testloader)
            val_step=0
            # test
            for proteins in t:

                self.optimizer.zero_grad()
                proteins=[proteins]


                x, aatypes, seq_masks, residue_indexs, gt_batchs = tied_features(
                    proteins, self.device, False if self.trainmodel == 'G' else True)

                if pre_aatype!=None:
                    aatypes=pre_aatype

                input = {'X': x, 'S': aatypes, 'mask': seq_masks, 'residue_idx': residue_indexs}

                loss, result = self.model.design(input, gt_batchs, **self.R_param, loss_factor=self.loss)
                #self.relax(result,proteins)
                #self.write_chis(**result,name=proteins[0]['domain_name'][0].decode('UTF-8'))

                acc.update([result['ca_lddt'], result['Angle_accs'][0], result['Angle_accs'][1],
                            result['Angle_accs'][2]
                               , result['Angle_accs'][3], result['Angle_accs'][4],
                            result['Angle_accs'][5],
                            result['Angle_accs'][6]])
                losses.update([result['bfloss'], result['ca_loss'], result['chi_loss'],
                               result['bb_angle_loss']])

                if val_step % self.vis['logging_per_update'] == 0:
                    logging.info(
                        "quanteron_loss:%.2f;ca_loss:%.2f;chi_angle_loss:%.2f;bb_angle_loss:%.2f;step %d；lr:%4f" % (
                            float(losses.avg(0)), float(losses.avg(1)), float(losses.avg(2)), float(losses.avg(3)),
                            self.global_step,
                            self.optimizer.param_groups[0]['lr']))

                    logging.info(
                        "ca_lddt:%.2f;Angle_accs1:%.2f;2:%.2f;3:%.2f;chi1:%.2f;chi2:%.2f;chi3:%.2f;chi4:%.2f;" % (
                            float(acc.avg(0)), float(acc.avg(1)), float(acc.avg(2)), float(acc.avg(3)),
                            float(acc.avg(4)), float(acc.avg(5)), float(acc.avg(6)), float(acc.avg(7)))
                    )
                val_step=val_step+1
            logging.info(
                    "quanteron_loss:%.2f;ca_loss:%.2f;chi_angle_loss:%.2f;bb_angle_loss:%.2f;step %d；lr:%4f" % (
                        float(losses.avg(0)), float(losses.avg(1)), float(losses.avg(2)), float(losses.avg(3)),
                        self.global_step,
                        self.optimizer.param_groups[0]['lr']))

            logging.info(
                    "ca_lddt:%.2f;Angle_accs1:%.2f;2:%.2f;3:%.2f;chi1:%.2f;chi2:%.2f;chi3:%.2f;chi4:%.2f;" % (
                        float(acc.avg(0)), float(acc.avg(1)), float(acc.avg(2)), float(acc.avg(3)),
                        float(acc.avg(4)), float(acc.avg(5)), float(acc.avg(6)), float(acc.avg(7)))
                )

    def write_chis(self,chis,name,**kwargs):
        chis=chis.tolist()
        with open('/home/junyu/PycharmProjects/PDHS/designPDB/relax/chi_'+name+'.txt','w') as f:
            f.writelines(json.dumps(chis))


    def relax(self,result,proteins):
        from model.network.feats import atom14_to_atom37
        from model.np import protein
        from model.network.relax_protein import relax_protein

        atom37=atom14_to_atom37(result['final_atom_positions'].squeeze(0),proteins[0])

        #ADD UNK atoms
        atom37=atom37+(1-proteins[0]['atom37_atom_exists'][:,:,None])*proteins[0]['all_atom_positions']

        result.update(
            {'final_atom_positions': atom37,
             'final_atom_mask': proteins[0]['atom37_atom_exists']}
        )
        chain_index=np.zeros_like(proteins[0]['aatype'],dtype=np.int)
        design_protein=protein.from_prediction(
            features=proteins[0],
            result=result,
            chain_index=chain_index
        )

        design_protein_str=protein.to_pdb(design_protein)
        unrelaxed_output_path= self.output_dir+'//PDHS_'+proteins[0]['domain_name'][0].decode('UTF-8')
        with open(unrelaxed_output_path+'_unrelaxed.pdb', 'w') as fp:
            fp.write(design_protein_str)


        output_directory=self.output_dir
        output_name=proteins[0]['domain_name'][0].decode('UTF-8')

        unrelaxed_output_path=unrelaxed_output_path+'_relaxed.pdb'
        logging.info(f"Running relaxation on {unrelaxed_output_path}...")
        relax_protein('CUDA', design_protein, output_directory, output_name)

        logging.info(f"designed Relaxed output written to {unrelaxed_output_path}...")
    def design(self):

        #load MODEL
        self.load_module(Repacker,self.R_param,self.R_Optimzer,self.R_scheduler,self.checkpoint['R_per_weight']) ##self.checkpoint['R_per_weight']
        # build dataset
        testloader = StructureDataset(self.dataset['designset'], **self.dataset)


        self.testloader = testloader

        self.global_step = self.R_scheduler['start_epoch'] * len(self.testloader)
        self.model.zero_grad()


        self.design_step(0)








if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')


    args=argsx()
    # args.just_valastrain=True
    # args.dataset['shuffle']=False
    # args.R_scheduler['scheduler']['kwargs']['epochs']=1
    # args.vis['logging_per_update']=1

    args.vis['logging_per_update']=10
    args.just_valastrain=False
    args.dataset['shuffle']=True
    args.R_scheduler['scheduler']['kwargs']['epochs']=10

    args.trainmodel='R'
    args.train_R()




