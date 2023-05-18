
import torch
import torch.nn as nn
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

from data import builder_for_AL
import tqdm
from model.utils import AverageMeter


import math

from model.network.Generator import Generator
from data.data_module import StructureDataset,StructureLoader,tied_features
import logging
# set_logger1()
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

from model.np.residue_constants import restypes_with_x


class argsx():
    def __init__(self):


        self.just_valastrain=False  # for debug using little data

        self.device='cuda'
        # trainmodel
        self.trainmodel='G'

        #data
        self.dataset={
            'trainset':'/home/junyu/PycharmProjects/PDHS/data/trainset.pkl',
            'testset':'/home/junyu/PycharmProjects/PDHS/data/testset.pkl',
            'max_length':1000,
            'batch_size': 1000,
            'shuffle':True
        }

        self.G_param={
            'node_features' : 256,
            'edge_features': 256,
            'hidden_dim':256,
            'num_encoder_layers':3,
            'augment_eps':0.1,
            'k_neighbors' :48,
            'dropout' : 0.1,
            'vocab' : len(restypes_with_x),
            'use_ESSE' : False,
            'use_tri':True,
            'use_rbf':True,
            'use_sse':False,
        }

        self.G_Optimzer={'optimizer':{
                'type': 'Adam',
                'kwargs': {
                    'lr': 0.0001,
                    'weight_decay': 0.0002
                }}}

        self.G_scheduler= {'scheduler':{
                'type': 'CosLR',
                'kwargs': {
                    'epochs': 50,
                    'initial_epochs': 0,
                    'warming_up_init_lr': 0.00001
                }},
                'start_epoch':0}

        self.vis= {
            'logging_per_update':10
        }

        self.checkpoint= {
            'save_path':'/home/junyu/PycharmProjects/PDHS/save/G/',
            'resume':True,
            'pre_weight':'/home/junyu/PycharmProjects/PDHS/save/3layers__mesh24_RBF_NOSSE_NOESEE_0.1_shift42.pt'
        }



    def load_G(self):
        # build G model
        G = Generator(**self.G_param).cuda()
        optimizer_G,scheduler_G= builder_for_AL.build_opti_sche(G, **self.G_Optimzer, **self.G_scheduler)

        # load checkpoints
        if self.checkpoint['resume']:
            checkpoint = torch.load(self.checkpoint['pre_weight'], map_location=self.device)
            checkpoint_model = checkpoint["model_state_dict"]
            model_dict = G.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint_model.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            G.load_state_dict(checkpoint_model)
            del model_dict

        # beginning to train

        logging.info("load generator")

        return G,optimizer_G,scheduler_G



    def load_dataset(self):

        if self.just_valastrain:
            testloader = StructureLoader(StructureDataset(self.dataset['testset'],**self.dataset),**self.dataset)

            self.trainloader=testloader
            self.testloader=testloader


        else:
            trainloader = StructureLoader(StructureDataset(self.dataset['trainset'], **self.dataset),**self.dataset)
            testloader = StructureLoader(StructureDataset(self.dataset['testset'], **self.dataset),
                                          **self.dataset)
            self.trainloader = trainloader
            self.testloader=testloader

    def train_step(self,epoch,G,optimizer_G,scheduler_G):


        # tools
        acc = AverageMeter(['recovery_acc'])
        losses = AverageMeter(['CE loss'])
        logging.info(60*'-' )
        logging.info(f"\n[Train] Start training epoch {epoch}", )
        logging.info(60*'-' )
        G.train()
        t = tqdm.tqdm(self.trainloader)

        if self.G_scheduler['scheduler']['type'] != 'function':
            if isinstance(scheduler_G, list):
                for item in scheduler_G:
                    item.step(epoch)
            else:
                scheduler_G.step(epoch)

        #train
        for proteins in t:
            optimizer_G.zero_grad()
            x, aatypes, seq_masks, residue_indexs=tied_features(proteins,self.device,False if self.trainmodel=='G' else True)
            loss_CE, recovery_acc = G(x, aatypes, seq_masks, residue_indexs,Index_embed=False, **self.G_param)

            loss_CE.backward()
            optimizer_G.step()
            acc.update(recovery_acc.item())
            losses.update(loss_CE.item())

            if self.global_step%self.vis['logging_per_update']==0:

                logging.info(
                    "CEloss:%.2f;recovery:%.3f;step %d；lr:%4f" % (
                        float(losses.avg(0)),  float(acc.avg(0)),
                         self.global_step,
                        optimizer_G.param_groups[0]['lr']))

            self.global_step=self.global_step+1
        self.val_step(epoch,G)
        return G,optimizer_G,scheduler_G

    def val_step(self, epoch,model):
        with torch.no_grad():
            # tools
            acc = AverageMeter(['recovery_acc'])
            losses = AverageMeter(['CE loss'])
            logging.info(f"\n[VAL] Start val epoch {epoch}", )
            model.eval()
            t = tqdm.tqdm(self.testloader)


            val_step=0
            # test
            for proteins in t:
                x, aatypes, seq_masks, residue_indexs = tied_features(proteins, self.device,False if self.trainmodel == 'G' else True)
                loss_CE, recovery_acc = model(x, aatypes, seq_masks, residue_indexs,
                                               Index_embed=False, **self.G_param)


                acc.update(recovery_acc.item())
                losses.update(loss_CE.item())

                if val_step % self.vis['logging_per_update'] == 0:
                    logging.info(
                        "CEloss:%.2f;recovery:%.3f;step %d；" % (
                            float(losses.avg(0)), float(acc.avg(0)),
                            val_step,
                           ))
                val_step=val_step+1

    def save_checkpoints(self,epoch,G,optimizer_G):
        prefix=self.trainmodel+str(self.G_param['augment_eps'])+'_'+str(epoch)
        torch.save({
            'base_model': G.state_dict(),
            'optimizer': optimizer_G.state_dict(),

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

    def train_G(self,  ):

        G,optimizer_G,scheduler_G=self.load_G()


        # beginning to train

        G.zero_grad()

        # build dataset
        self.load_dataset()

        self.global_step=0



        for epoch in range(self.G_scheduler['start_epoch'], self.G_scheduler['scheduler']['kwargs']['epochs'] + 1):
            G,optimizer_G,scheduler_G=self.train_step(epoch,G,optimizer_G,scheduler_G)
            self.save_checkpoints(epoch,G,optimizer_G)
            # self.val_step(epoch,G)

    def eval(self):


        G, optimizer_G, scheduler_G = self.load_G()

        # beginning to train

        G.zero_grad()

        # build dataset
        self.just_valastrain=True  # for debug using little data
        self.load_dataset()

        self.global_step = 0

        self.val_step(epoch=0,model=G)







if __name__ == "__main__":
    args=argsx()
    # args.train_G()
    args.eval()
