import glob

import numpy as np
import torch
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import pickle
from data import builder_for_AL
import tqdm
from model.utils import AverageMeter
import pandas

from model.network.Generator import Generator
from model.network.Joint_model import Joint_module
from data.data_module import StructureDataset,StructureLoader,tied_features_J,gt_batch
import logging
# set_logger1()
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

from model.np.residue_constants import restypes_with_x,restypes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class argsx():
    def __init__(self):


        self.just_valastrain=True  # for debug using little data

        self.device=device

        # trainmodel
        self.trainmodel='J'

        self.loss={
            'aux_f':0.1,
            'fape_f':1, # it begian from 16

            'aa_f': 2,
            'angle_f':1,
            'chi_weight' : 0,
            'bb_torsion_weight' : 1,
            'angle_norm_weight' : 0.1,
            'aatempweight':0.1,
            'kind':len(restypes)
        }

        #data
        self.dataset={
            'trainset':'/home/jorey/pdhs/dataset//S40_train_fbb_fix256.pkl',
            'testset':'//home/jorey/pdhs/dataset//S40_test_fbb_fix256.pkl',
            'designset': '/home/junyu/PycharmProjects/pdhs/Testset/66Class.pkl',
            'max_length':1000,
            'batch_size': 2000,
            'shuffle':True
        }

        self.G_param={
            'node_features' : 256,
            'edge_features': 256,
            'hidden_dim':256,
            'num_encoder_layers':3,
            'augment_eps':0.01,
            'k_neighbors' :48,
            'dropout' : 0.1,
            'vocab' : len(restypes_with_x),
            'use_ESSE' : False,
            'use_tri':True,
            'use_rbf':True,
            'use_sse':False,
            'use_Eaatype': False,
            'log_softmax':True,
            'msas':6
        }

        eps=1e-8
        self.J_param={
            'n_module_str':2,
            'msa_layer':12,
            'IPA_layer':12,
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
            "no_resnet_blocks": 2,
            "no_angles": 7,
            "trans_scale_factor": 10.,
            "a_epsilon": 1e-12,  # 1e-12,
            'noiseFactor':2,
            'loss_factor':self.loss


        }

        self.G_Optimzer={'optimizer':{
                'type': 'Adam',
                'kwargs': {
                    'lr': 0.0001,
                    'weight_decay': 0.0002
                }}}
        self.J_Optimzer=self.G_Optimzer

        self.G_scheduler= {'scheduler':{
                'type': 'CosLR',
                'kwargs': {
                    'epochs': 1,
                    'initial_epochs': 0,
                    'warming_up_init_lr': 0.00001
                }},
                'start_epoch':0}
        self.J_scheduler=self.G_scheduler


        self.vis= {
            'logging_per_update':10
        }

        self.checkpoint= {
            'save_path':'/home/omnisky/data/everyOne/yjy/pdhs/save/',
            'resume_G':False,
            'G_per_weight':'/home/omnisky/data/everyOne/yjy/pdhs/save/G/3layers__mesh24_RBF_NOSSE_NOESEE_0.1_shift42.pt',
            'J_per_weight': '/home/jorey/pdhs/save/J/Gw0.1_msa6_6*2_1_4.pth'
        }


        print(self.loss)
        print(self.dataset)
        print(self.G_param)
        print(self.J_param)
        print(self.checkpoint)







    def load_module(self,module,module_param,Optimzer_param,scheduler_param,Resume_pth=None):
        # build G model
        model = module(**module_param,G_param=self.G_param)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(model)  # device_ids will include all GPU devices by default
        else:
            self.model = model
        self.model = self.model.to(device)

        self.optimizer, self.scheduler = builder_for_AL.build_opti_sche(model, **Optimzer_param, **scheduler_param)

        # load checkpoints
        if Resume_pth is not None:
            checkpoint_path=Resume_pth
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            checkpoint_model = checkpoint["base_model"]  #base_model  model_state_dict
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint_model.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(checkpoint_model)
            #self.optimizer.load_state_dict(checkpoint["optimizer"])
            del model_dict

        #load G
        if self.checkpoint['resume_G']:
            Gcheckpoint = torch.load(self.checkpoint['G_per_weight'], map_location=self.device)
            checkpoint_model = Gcheckpoint["model_state_dict"]
            model_dict = model.state_dict()
            pretrained_dict = {'G.' + k: v for k, v in checkpoint_model.items() if 'G.' + k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        # beginning to train

        logging.info("load model checkpoints")




    def load_dataset(self):

        if self.just_valastrain:
            testloader = StructureDataset(self.dataset['testset'],**self.dataset)   # StructureLoader(  ,**self.dataset)

            self.trainloader=testloader
            self.testloader=testloader


        else:
            trainloader = StructureLoader(StructureDataset(self.dataset['trainset'], **self.dataset),**self.dataset)
            testloader = StructureLoader(StructureDataset(self.dataset['testset'], **self.dataset),
                                          **self.dataset)
            self.trainloader = trainloader
            self.testloader=testloader

    def train_step(self,epoch):

        # tools
        acc = AverageMeter(
            ['Grecovery', 'Frecovery', 'Ca_lddt', 'angles_acc1', 'angles_acc2', 'angles_acc3'])
        losses = AverageMeter(['FCE loss','fape loss','bb angles loss','ca_rmsd_loss'])
        logging.info(60*'-' )
        logging.info(f"\n[Train] Start training epoch {epoch}", )
        logging.info(60*'-' )
        self.model.train()
        t = tqdm.tqdm(self.trainloader)


        self.scheduler.step(epoch)

        #train
        for proteins in t:
            self.optimizer.zero_grad()


            gpus=torch.cuda.device_count()
            batchs=int(len(proteins)/gpus)

            if batchs!=0:
                proteins=proteins[:int(batchs*gpus)]
            else:
                diff=gpus-len(proteins)%gpus
                for i in range(diff):
                    proteins.append( proteins[-1])

            x, aatypes, seq_masks, residue_indexs,torsion_angles,alt_torsion_angles,torsion_angle_masks=tied_features_J(
                proteins,self.device,False if self.trainmodel=='G' else True)
            gt_batchs=gt_batch(x, aatypes, seq_masks, torsion_angles,torsion_angle_masks,self.J_param['r_epsilon'],self.device)

            input = {'X': x,'S': aatypes,'mask': seq_masks,'residue_idx': residue_indexs}


            loss,seq_recovery_rate_g,result=self.model(input,gt_batchs)
            loss=torch.mean(loss)
            seq_recovery_rate_g=torch.mean(seq_recovery_rate_g)

            loss.backward()
            self.optimizer.step()

            acc.update([seq_recovery_rate_g.item(), torch.mean(result['recovery']).item(), torch.mean(result['ca_lddt']).item(),
                        torch.mean(result['Angle_accs']).item(),
                        ])
            losses.update([torch.mean(result['aa_loss']).item(), torch.mean(result['bb_fape_loss']).item(),
                           torch.mean(result['bb_angle_loss']).item(),torch.mean(result['ca_rmsd_loss']).item()])

            if self.global_step%self.vis['logging_per_update']==0:
                logging.info(
                    "CEloss:%.2f;bb_fape_loss:%.2f;bb_angle_loss:%.2f;ca_rmsd_loss:%.2f;step %d；lr:%4f" % (
                        float(losses.avg(0)), float(losses.avg(1)),
                        float(losses.avg(2)), float(losses.avg(3)),
                        self.global_step,
                        self.optimizer.param_groups[0]['lr']))

                logging.info(
                    "GRe:%.2f;FRe:%.2f;ca_lddt:%.2f;Angle_accs:%.2f;" % (
                        float(acc.avg(0)), float(acc.avg(1)), float(acc.avg(2)), float(acc.avg(3)),
                    ))
            self.global_step=self.global_step+1

    def val_step(self, epoch):
        with torch.no_grad():
            # tools
            acc = AverageMeter(['Grecovery', 'Frecovery', 'Ca_lddt', 'angles_acc1', 'angles_acc2', 'angles_acc3'])
            losses = AverageMeter(['FCE loss', 'fape loss',  'bb angles loss','ca_rmsd_loss'])
            logging.info(f"\n[VAL] Start val epoch {epoch}", )
            self.model.eval()
            t = tqdm.tqdm(self.testloader)
            val_step=0
            # test

            namelist=[]
            recoverlist=[]

            cenlist=[]
            predaatype=[]
            gtaatype=[]

            for proteins in t:
                self.optimizer.zero_grad()

                gpus = torch.cuda.device_count()
                batchs = int(len(proteins) / gpus)

                # if batchs != 0:
                #     proteins = proteins[:int(batchs * gpus)]
                # else:
                #     diff = gpus - len(proteins) % gpus
                #     for i in range(diff):
                #         proteins.append(proteins[-1])
                proteins=[proteins]
                x, aatypes, seq_masks, residue_indexs, torsion_angles, alt_torsion_angles, torsion_angle_masks = tied_features_J(
                    proteins, self.device, False if self.trainmodel == 'G' else True)
                gt_batchs = gt_batch(x, aatypes, seq_masks, torsion_angles, torsion_angle_masks,
                                     self.J_param['r_epsilon'], self.device)

                input = {'X': x, 'S': aatypes, 'mask': seq_masks, 'residue_idx': residue_indexs}

                loss, seq_recovery_rate_g, result = self.model(input, gt_batchs)

                seq_recovery_rate_g = torch.mean(seq_recovery_rate_g)




                acc.update([seq_recovery_rate_g.item(), torch.mean(result['recovery']).item(),
                            torch.mean(result['ca_lddt']).item(),
                            torch.mean(result['Angle_accs']).item(),
                            ])
                losses.update([torch.mean(result['aa_loss']).item(), torch.mean(result['bb_fape_loss']).item(),
                               torch.mean(result['bb_angle_loss']).item(), torch.mean(result['ca_rmsd_loss']).item()])


                # namelist.append(proteins[0]['domain_name'][0].decode('UTF-8'))
                # cenlist=cenlist+proteins[0]['cens'].tolist()
                # predaatype=predaatype+np.asarray(aatype[0].detach().cpu()).tolist()
                # gtaatype=gtaatype+np.asarray(aatypes[0].detach().cpu()).tolist()

                # recoverlist.append(float(result['recovery'].item()))

                # testserrecovery=pandas.DataFrame([cenlist,predaatype])



                if val_step % self.vis['logging_per_update'] == 0:
                    logging.info(
                        "CEloss:%.2f;bb_fape_loss:%.2f;bb_angle_loss:%.2f;ca_rmsd_loss:%.2f;step %d；lr:%4f" % (
                            float(losses.avg(0)), float(losses.avg(1)),
                            float(losses.avg(2)),float(losses.avg(3)),
                            self.global_step,
                            self.optimizer.param_groups[0]['lr']))

                    logging.info(
                        "GRe:%.2f;FRe:%.2f;ca_lddt:%.2f;Angle_accs1:%.2f;" % (
                            float(acc.avg(0)), float(acc.avg(1)), float(acc.avg(2)), float(acc.avg(3))
                        ))
                val_step=val_step+1

            testserrecovery=pandas.DataFrame([cenlist,predaatype,gtaatype]).transpose()
            testserrecovery.columns=['cen','paa','gtaa']
            print(float(acc.avg(1)))
            f = open('../images/recovry_vio/aatype_cens_casp15', 'wb')
            pickle.dump(testserrecovery, f)
            f.close()

    def train_G_step(self,epoch):

        # tools
        acc = AverageMeter(
            ['Grecovery', 'Frecovery', 'Ca_lddt', 'angles_acc1', 'angles_acc2', 'angles_acc3'])
        losses = AverageMeter(['FCE loss','fape loss','bb angles loss'])
        logging.info(60*'-' )
        logging.info(f"\n[Train] Start training epoch {epoch}", )
        logging.info(60*'-' )
        self.model.train()
        t = tqdm.tqdm(self.trainloader)


        self.scheduler.step(epoch)

        #train
        for proteins in t:
            self.optimizer.zero_grad()

            x, aatypes, seq_masks, residue_indexs=tied_features_J(
                proteins,self.device,False if self.trainmodel=='G' else True)


            input = {'X': x,'S': aatypes,'mask': seq_masks,'residue_idx': residue_indexs}


            loss,seq_recovery_rate_g=self.model(**input,log_softmax=True,TrainG=True)

            loss.backward()
            self.optimizer.step()

            acc.update([seq_recovery_rate_g.item()])
            losses.update([loss.detach().cpu().numpy()])

            if self.global_step%self.vis['logging_per_update']==0:
                logging.info(
                    "CEloss:%.5f;acc:%.2f;step:%.2f;lr:%.2f;" % (
                        float(losses.avg(0)), float(acc.avg(0)),

                        self.global_step,
                        self.optimizer.param_groups[0]['lr']))

            self.global_step=self.global_step+1

    def VAL_G_step(self, epoch):
        with torch.no_grad():
            # tools
            acc = AverageMeter(
                ['Grecovery', 'Frecovery', 'Ca_lddt', 'angles_acc1', 'angles_acc2', 'angles_acc3'])
            losses = AverageMeter(['FCE loss', 'fape loss', 'bb angles loss'])
            logging.info(60 * '-')
            logging.info(f"\n[Train] Start training epoch {epoch}", )
            logging.info(60 * '-')
            self.model.train()
            t = tqdm.tqdm(self.testloader)

            self.scheduler.step(epoch)

            # train
            for proteins in t:
                self.optimizer.zero_grad()

                x, aatypes, seq_masks, residue_indexs = tied_features_J(
                    proteins, self.device, False if self.trainmodel == 'G' else True)

                input = {'X': x, 'S': aatypes, 'mask': seq_masks, 'residue_idx': residue_indexs}

                loss, seq_recovery_rate_g = self.model(**input, log_softmax=True, TrainG=True)



                acc.update([seq_recovery_rate_g.item()])
                losses.update([loss.detach().cpu().numpy()])

                if self.global_step % self.vis['logging_per_update'] == 0:
                    logging.info(
                        "CEloss:%.5f;acc:%.2f;step:%.2f;lr:%.2f;" % (
                            float(losses.avg(0)), float(acc.avg(0)),

                            self.global_step,
                            self.optimizer.param_groups[0]['lr']))

                self.global_step = self.global_step + 1

            logging.info(
                "CEloss:%.5f;acc:%.2f;step:%.2f;lr:%.2f;" % (
                    float(losses.avg(0)), float(acc.avg(0)),

                    self.global_step,
                    self.optimizer.param_groups[0]['lr']))


    def save_checkpoints(self,epoch):
        prefix=self.trainmodel+str('/J_0.1_hardF_midloss_Single_Noise')+str(self.J_param['noiseFactor'])+'_'+str(epoch)
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

    def train_J(self,  ):

        #load MODEL
        self.load_module(Joint_module,self.J_param,self.J_Optimzer,self.J_scheduler,None) ## self.checkpoint['J_per_weight']

        # build dataset
        self.load_dataset()

        self.global_step=self.G_scheduler['start_epoch']*len(self.trainloader)
        self.model.zero_grad()

        for epoch in range(self.G_scheduler['start_epoch'], self.G_scheduler['scheduler']['kwargs']['epochs'] + 1):
            # self.train_step(epoch)
            # self.save_checkpoints(epoch)
            self.val_step(epoch)


    def train_G(self,  ):

        #load MODEL
        self.load_module(Generator,self.G_param,self.G_Optimzer,self.G_scheduler,self.checkpoint['G_per_weight']) ## self.checkpoint['G_per_weight']

        # build dataset
        self.load_dataset()

        self.global_step=self.G_scheduler['start_epoch']*len(self.trainloader)
        self.model.zero_grad()

        for epoch in range(self.G_scheduler['start_epoch'], self.G_scheduler['scheduler']['kwargs']['epochs'] + 1):
            self.train_G_step(epoch)
            self.save_checkpoints(epoch)
            self.VAL_G_step(epoch)

    def design_step(self, epoch):
        with torch.no_grad():
            # tools
            acc = AverageMeter(['Grecovery', 'Frecovery', 'Ca_lddt', 'angles_acc1', 'angles_acc2', 'angles_acc3'])
            losses = AverageMeter(['FCE loss', 'q loss', 'ca loss', 'bb angles loss'])
            logging.info(f"\n[VAL] Start val epoch {epoch}", )
            self.model.eval()
            t = tqdm.tqdm(self.testloader)
            val_step=0
            # test

            with open('design_6layers_test6_G0.1.fa', 'w') as f:
                f.writelines(str(self.checkpoint))
                f.writelines('\n')
                f.writelines(str(self.J_param))
                f.writelines('\n')
                f.writelines(str(self.G_param))
                f.writelines('\n')
                f.writelines(40*'-')
                f.writelines('\n')

                for proteins in t:
                    self.optimizer.zero_grad()
                    print('now design',proteins['domain_name'])
                    x, aatypes, seq_masks, residue_indexs, torsion_angles, alt_torsion_angles, torsion_angle_masks = tied_features(
                        [proteins], self.device, False if self.trainmodel == 'G' else True)
                    gt_batchs = gt_batch(x, aatypes, seq_masks, torsion_angles, torsion_angle_masks,
                                         self.J_param['r_epsilon'])

                    input = {'X': x, 'S': aatypes, 'mask': seq_masks, 'residue_idx': residue_indexs}

                    loss, seq_recovery_rate_g, result = self.model.design(input, gt_batchs, **self.J_param, loss_factor=self.loss)

                    acc.update([seq_recovery_rate_g.item(), result['recovery'].item(), result['ca_lddt'].item(),
                                result['Angle_accs'][0].item(), result['Angle_accs'][1].item(),
                                result['Angle_accs'][2].item()
                                ])
                    losses.update([result['aa_loss'].item(), result['quanteron_loss'].item(), result['ca_loss'].item(),
                                   result['bb_angle_loss'].item()])

                    desingseq=result['design']
                    head='>design|'+str(proteins['domain_name'][0])[2:8]+'|Recovery '+str(result['recovery'])+'|CA Lddt '+str(result['ca_lddt'])+'|Length '+str(len(desingseq))

                    line1='>Native|'+str(proteins['seq'][0])[2:-1]
                    line2='>Design|'+desingseq
                    f.write(head)
                    f.writelines('\n')
                    f.write(line1)
                    f.writelines('\n')
                    f.write(line2)
                    f.writelines('\n')
                    f.writelines('\n')



                    if val_step % self.vis['logging_per_update'] == 0:
                        logging.info(
                            "CEloss:%.2f;quanteron_loss:%.2f;ca_loss:%.2f;bb_angle_loss:%.2f;step %d；lr:%4f" % (
                                float(losses.avg(0)), float(losses.avg(1)), float(losses.avg(2)),
                                float(losses.avg(3)),
                                self.global_step,
                                self.optimizer.param_groups[0]['lr']))

                        logging.info(
                            "GRe:%.2f;FRe:%.2f;ca_lddt:%.2f;Angle_accs1:%.2f;2:%.2f;3:%.2f;" % (
                                float(acc.avg(0)), float(acc.avg(1)), float(acc.avg(2)), float(acc.avg(3)),
                                float(acc.avg(4)), float(acc.avg(5))
                            ))

                    val_step=val_step+1
    def design(self):

        self.load_module(Joint_module, self.J_param, self.J_Optimzer, self.J_scheduler,
                         self.checkpoint['J_per_weight'])  ##
        # build dataset
        testloader = StructureDataset(self.dataset['designset'], **self.dataset)


        self.testloader = testloader

        self.global_step = self.G_scheduler['start_epoch'] * len(self.testloader)
        self.model.zero_grad()

        self.vis['logging_per_update'] =10
        self.design_step(0)




def fromrepackerdatta_tofbb():
    pickles='/home/jorey/pdhs/dataset/S40_fbb_train_bfssecenres_fix256_5.pkl'
    pkl_file = open(pickles, 'rb')
    dataset = pickle.load(pkl_file)




    for i in tqdm.tqdm(range(len(dataset))):
        del dataset[i]['rigidgroups_gt_frames']
        del dataset[i]['rigidgroups_gt_exists']
        del dataset[i]['rigidgroups_alt_gt_frames']
        del dataset[i]['atom14_gt_positions']
        del dataset[i]['atom14_alt_gt_positions']
        del dataset[i]['atom14_gt_exists']
        del dataset[i]['atom14_alt_gt_exists']
        del dataset[i]['atom14_atom_exists']
        del dataset[i]['atom14_atom_is_ambiguous']
        # del dataset[i]['residx_atom14_to_atom37']

    output = open('/home/jorey/pdhs/dataset/S40_fbb_train_bfssecenres_fix256_5.pkl', 'wb')

    # 写入到文件
    pickle.dump(dataset, output)
    output.close()

def add_all_pkl():
    pickles=glob.glob('/home/jorey/pdhs/dataset/S40_fbb_train_bfssecenres_fix256.pkl')

    data=[]
    for i in pickles:
        pkl_file = open(i, 'rb')
        dataset = pickle.load(pkl_file)
        for line in dataset:
            L_max = 256
            # get backbone atoms  # [L, 4, 3]
            x_pad = np.pad(line['backbone_atom_positions'][:, :4, :], [[0, L_max - line['backbone_atom_positions'].shape[0]], [0, 0], [0, 0]],
                            'constant', constant_values=-0.)
            data.append(x_pad)


    output = open('/home/jorey/pdhs/dataset/S40_train_bb.pkl', 'wb')
    # 写入到文件
    pickle.dump(data, output)
    output.close()


if __name__ == "__main__":
    # add_all_pkl()

    args=argsx()
    args.just_valastrain=True
    args.dataset['shuffle']=False
    #
    # # args.just_valastrain=False
    # # args.dataset['shuffle']=True
    #
    # args.trainmodel='J'
    # #args.train_G()
    #
    args.train_J()




