import torch
import torch.nn as nn
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
from CLIP.data import builder_for_AL

import tqdm
import time
from CLIP.model.utils import AverageMeter
from CLIP.model.metrics import Metrics

import math
from CLIP.model.network.Joint_model import Joint_module
from set_logger import set_logger1
import logging
set_logger1()
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
import json
def get_temp(config, niter):
    if config.get('temp') is not None:
        start = config.temp.start
        target = config.temp.target
        ntime = config.temp.ntime
        if niter > ntime:
            return target
        else:
            temp = target + (start - target) * (1. + math.cos(math.pi * float(niter) / ntime)) / 2.
            return temp
    else:
        return 0

def _check_msas_exist(names,check_name,get_name,miss_name):


    # get_name=[]
    # miss_name=[]


    get=0
    check_name = check_name.lower()
    check_name_f, check_name_b = check_name.split('.')  # [0]

    for i in range(len(names)):
        x=names[i]
        thisname=names[i]
        thisname=thisname.lower()
        fname=thisname[1:5]
        try:
            bname = thisname[5]
        except:
            print(thisname)


        if fname==check_name_f and bname==check_name_b:
            get_name.append(check_name)
            get=1
            break
    if get==0:
        #print('we miss '+str(check_name))
        miss_name.append(check_name)


    return    get_name,miss_name



I2Sdicts={30: '-', 1: '*', 4: 'L', 10: 'R', 19: 'Y', 14: 'P', 5: 'A', 7: 'V', 6: 'G', 12: 'I', 21: 'H', 18: 'F', 8: 'S', 22: 'W', 9: 'E', 11: 'T', 20: 'M', 13: 'D', 16: 'Q', 15: 'K', 23: 'C', 17: 'N'}

#[!!!] actually 30 is missing_mask, 1 is padding mask

class argsx():
    def __init__(self):
        self.resume =True
        self.muliti_GPU=False

        self.experiment_path='/home/junyu/桌面/code/ProteinCLIP/CLIP/save/'
        self.config='Clip.yaml'

        self.val_freq=1
        self.data='CATH4.3'


        self.Generator_checkpoint_path = self.experiment_path + '/6layersmeshmpnnsse_43bad_trans_noise0.1_shift67' + '.pt'
        self.Joint_checkpoint_path = self.experiment_path + 'G_0.1k1_RMSD_loss_aux_8msafirst__IPA_33_4recyle_10noise_Norm_chemNoloss_1-last' + '.pth'

        self.eval_mode=False

        self.just_valastrain=False  # for debug using little data
        self.backbone_noise=0.1
        self.Joint_Factor=10
        self.auxk=0.02

        self.n_module_str=4
        self.n_layer=3
        self.IPA_layer=3

        #Generator
        self.generator_layer=6
        self.use_tri=True
        self.use_rbf=True
        self.use_sse=True
        self.use_ESSE=False




def train_CLIPnet(args, config, ):
    '''
    fix generator, max kl of generator's seq and real seq
    '''
    # vis params
    logging.info(40 * '-')
    logging.info( f'now fix generator, max kl of generator,s seq and real seq ')
    logging.info(40 * '-')

    # cuda
    if args.muliti_GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    else:
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")





    #build CLIP model

    SEQGenerator = builder_for_AL.model_builder(args,config.model,model='SEQGenerator',backbone_noise=args.backbone_noise,)
    Joint_OPT =Joint_module(n_module_str=args.n_module_str, n_layer=args.n_layer,IPA_layer=args.IPA_layer ,)#builder_for_AL.model_builder(args,config.model,model='SEQCLIP_GPT') # #
    if args.muliti_GPU:
        SEQGenerator = nn.DataParallel(SEQGenerator)
    SEQGenerator=SEQGenerator.cuda()
    Joint_OPT=Joint_OPT.cuda()


    optimizer_Joint ,scheduler_Joint = builder_for_AL.build_opti_sche(Joint_OPT, config)
    optimizer_G, scheduler_G = builder_for_AL.build_opti_sche(SEQGenerator, config)


    #load checkpoints
    if args.resume:
        checkpoint = torch.load(args.Generator_checkpoint_path, map_location=device)
        checkpoint_model=checkpoint["model_state_dict"]
        model_dict = SEQGenerator.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint_model.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        SEQGenerator.load_state_dict(model_dict)


        checkpoint = torch.load(args.Joint_checkpoint_path, map_location=device)
        checkpoint_model=checkpoint["base_model"]
        Joint_model_dict = Joint_OPT.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint_model.items() if k in Joint_model_dict}
        Joint_model_dict.update(pretrained_dict)
        Joint_OPT.load_state_dict(Joint_model_dict)


        # del Joint_model_dict
        del model_dict



    #beginning to train
    start_epoch = 0
    logging.info("entring epochs")

    SEQGenerator.eval()

    Joint_OPT.zero_grad()
    SEQGenerator.zero_grad()

    # build dataset
    just_val = args.just_valastrain
    if just_val:
        train_dataloader = builder_for_AL.dataset_builder(args, config, justval=just_val, justtest=False,
                                                          dataset='CATH4.3')
        val_dataloader = train_dataloader

    else:
        train_dataloader, val_dataloader = builder_for_AL.dataset_builder(args, config, justval=just_val,
                                                                          justtest=False, dataset='CATH4.3')



    for epoch in range(start_epoch, config.max_epoch + 1):
        logging.info(f"\n[Train] Start training epoch {epoch}", )
        Joint_OPT.train()  # set model to training mode

        # load data
        t = tqdm.tqdm(train_dataloader)

        #tools
        acc = AverageMeter([ 'recovery_acc','recovery_acc_msa'])
        losses = AverageMeter(['CE loss','kl loss','dis'])

        num_iter = 0
        n_batches = len(train_dataloader)
        idx=0

        msa_num =8



        if config.scheduler.type != 'function':
            if isinstance(scheduler_Joint, list):
                for item in scheduler_Joint:
                    item.step(epoch)
            else:
                scheduler_Joint.step(epoch)


        for proteins in t:
            num_iter += 1
            n_itr = epoch * n_batches + idx
            # pre_deal data; get points
            points, real_tokens, residue_idx, mask,seq_for_conveter,SSE3_seq,SSE8_seq = builder_for_AL.tied_featurize(proteins, device)  #B,L

            optimizer_Joint.zero_grad()
            optimizer_G.zero_grad()

            loss_CE_all=0
            recovery_acc_all=0
            faketokenlist=[]


            # time1=time.time()
            for i in range(msa_num-1):
                with torch.no_grad():

                    loss_CE, recovery_acc, fake_token, h_V ,_= SEQGenerator(points, real_tokens, mask, residue_idx, SSE3_seq,
                                                                          SSE8_seq, Index_embed=True, use_gradient=False,
                                                                            use_tri=args.use_tri,use_rbf=args.use_rbf,use_sse=args.use_sse)
                    loss_CE, recovery_acc, fake_token, h_V, _ = SEQGenerator(points, real_tokens, mask, residue_idx,
                                                                                     SSE3_seq,
                                                                                     SSE8_seq, Index_embed=True,
                                                                                     use_gradient=False)


                faketokenlist.append(fake_token)
                loss_CE_all = loss_CE_all + loss_CE
                recovery_acc_all = recovery_acc_all + recovery_acc

            faketokenlist = torch.stack(faketokenlist, dim=1)

            loss_CE = loss_CE_all / msa_num
            recovery_acc = recovery_acc_all / msa_num


            loss_CE_msa, recovery_acc_msa, dis ,pred= Joint_OPT(faketokenlist,real_tokens,  points,  mask,residue_idx,
                                                             args.Joint_Factor,args.auxk)

            k=1
            lamb=10
            if idx%1==0:
                Joint_loss =loss_CE_msa+k*dis
                Joint_loss.backward()
                optimizer_Joint.step()

                del faketokenlist
                del points

            else:
                optimizer_G.zero_grad()
                # Generator_loss,kld_weight = compute_loss(loss_CE,torch.exp(fake+1),config, n_itr)
                Generator_loss =loss_CE_msa+k*dis
                Generator_loss.backward()
                optimizer_G.step()


            # update records
            losses.update([loss_CE.item(), loss_CE_msa.item(),dis.item()])
            acc.update([recovery_acc.item(),recovery_acc_msa.item()])



            idx = idx + 1
            if num_iter == config.logging_per_update:
                num_iter = 0
                logging.info(
                    "CEloss:%.2f;msa loss:%.2f;recovery:%.3f;recoverymsa:%.3f;dis %f；step %d；lr:%4f" % (
                        float(losses.avg(0)), float(losses.avg(1)),float(acc.avg(0)),
                        float(acc.avg(1)), float(losses.avg(2)), n_itr,
                        optimizer_Joint.param_groups[0]['lr']))




        #     # val
        # builder_for_AL.save_checkpoint(SEQGenerator, optimizer_G, epoch,
        #                                'test6layer_21dim_32Pdgx_GPT' + '-last',
        #                                args, )
        builder_for_AL.save_checkpoint(Joint_OPT, optimizer_Joint, epoch,
                                       'G_'+str(args.backbone_noise)+'k'+str(k)+'_'+'RMSD_loss_aux'+'_'+str(msa_num)+'msafirst__IPA_33_4recyle_'+str(args.Joint_Factor)+'noise_Norm_chemNoloss_'+str(epoch) + '-last',
                                       args, )
        if epoch % args.val_freq == 0:
            # Validate the current model

            with torch.no_grad():
                validate_trainCLIP(SEQGenerator, Joint_OPT, val_dataloader, epoch, config, ((1 + epoch) * n_batches),msa_num,k)

    torch.cuda.empty_cache()





def validate_trainCLIP(SEQGenerator,SEQCLIP, test_dataloader, epoch,  config, n_itr,msanum,k):
    logging.info(f"\n[VALIDATION] Start CLIP validating epoch {epoch}", )
    SEQCLIP.eval()

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    acc = AverageMeter([ 'recovery_acc','recovery_acc'])
    t = tqdm.tqdm(test_dataloader)
    test_losses = AverageMeter(['CE loss','clip loss','RMSDloss'])

    num_iter = 0


    idx = 0
    for proteins in t:
        num_iter += 1
        n_itr = n_itr
        # pre_deal data; get points
        points, real_tokens, residue_idx, mask, seq_for_conveter,SSE3_seq,SSE8_seq = builder_for_AL.tied_featurize(proteins,device)  # B,L
        real_tokens[mask==0]=0  # pading and miss is 0


        pair=0
        faketokenlist=[]
        loss_CE_all=0
        recovery_acc_all=0

        for i in range(msanum):
            loss_CE,recovery_acc,fake_token,h_V,_=SEQGenerator(points,real_tokens,mask,residue_idx,SSE3_seq,SSE8_seq,Index_embed=True,use_gradient=False)




            faketokenlist.append(fake_token)
            loss_CE_all=loss_CE_all+loss_CE
            recovery_acc_all=recovery_acc_all+recovery_acc


        faketokenlist=torch.stack(faketokenlist,dim=1)

        loss_CE=loss_CE_all/msanum
        recovery_acc=recovery_acc_all/msanum
        #pair=pair/msanum
        loss_CE_msa, recovery_acc_msa, dis ,pred= SEQCLIP(faketokenlist, real_tokens, points, mask, residue_idx,
                                                       args.Joint_Factor,args.auxk)


        CLIP_loss=loss_CE_msa+k*dis
        # update records
        test_losses.update([loss_CE.item(), CLIP_loss.item(),dis.item()])
        acc.update([recovery_acc.item(),recovery_acc_msa.item()])


        idx = idx + 1
        if num_iter == config.logging_per_update:
            num_iter = 0
            t.set_description("CE loss: %.5f;dis: %.5f;r1: %.5f;r2: %.5f;global_step %d；" % (
                float(test_losses.avg(0)),float( test_losses.avg(2)),  float(acc.avg(0)),float(acc.avg(1)), n_itr))

    # ax = plt.figure().add_subplot(111, projection='3d')
    # ax.scatter(np.asarray(x),np.asarray(y),np.asarray(c),c=np.asarray(c))
    # plt.show()


def compute_loss(loss_1, loss_2, config, niter):
    '''
    compute the final loss for optimization
    For dVAE: loss_1 : reconstruction loss, loss_2 : kld loss
    '''
    start = config.kldweight.start
    target = config.kldweight.target
    ntime = config.kldweight.ntime

    _niter = niter - 20000
    if _niter > ntime:
        kld_weight = target
    elif _niter < 0:
        kld_weight = 0.1
    else:
        kld_weight = target + (start - target) * (1. + math.cos(math.pi * float(_niter) / ntime)) / 2.


    loss = loss_1 + kld_weight * loss_2

    return kld_weight


import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
if __name__ == "__main__":
    args=argsx()
    config = builder_for_AL.cfg_from_yaml_file(args.config )


    ####init Generator
    # run_Generatornet(args,config)

    ####alter train
    #
    # logging.info(f"------------------------run the th CLIP -----------------------")
    train_CLIPnet(args,config)
    # logging.info(f"------------------------run the {i} th train_Generatorby_CLIP -----------------------")
    # train_Generatorby_CLIP(args,config)
