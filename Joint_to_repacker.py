import glob
import json

import numpy as np
import torch

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
import ml_collections as mlc
import tqdm

import sys

from model.utils import AverageMeter
from model.network.Repacker import Repacker
from model.network.Joint_model import Joint_module
from data.data_module import StructureDataset,StructureLoader,tied_features,gt_batch
import logging

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

from model.np.residue_constants import restypes_with_x,restypes,restype_order,restype_order_with_x
import argparse

class argsx():
    def __init__(self):


        self.just_valastrain=True  # for debug using little data

        self.device='cuda'

        # trainmodel
        self.trainmodel='R'


        #data
        self.dataset={
            'designset': './design/CASP15_nocut.pkl',
            'max_length':1000,
            'batch_size': 1000,
            'shuffle':True
        }

        #output
        self.output_dir = '/output/'


        self.loss={
            'aux_f':0.1,
            'fae_t_f':1, # it begian from 16
            'fae_q_f': 1,
            'fape_f': 1,
            'angle_f':1,
            'chi_weight' : 4,
            'bb_torsion_weight' : 1,
            'angle_norm_weight' : 0.1,
            'aatempweight':0.1,
            'kind':len(restypes_with_x)
        }

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

        eps=1e-8
        self.J_param={
            'n_module_str':2,
            'msa_layer':6,
            'IPA_layer':6,
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
            'noiseFactor':0.5,
            'loss_factor':self.loss


        }

        self.G_param={
            'node_features' : 256,
            'edge_features': 256,
            'hidden_dim':256,
            'num_encoder_layers':3,
            'augment_eps':0.005,
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

            'device': self.device,
            'pairwise_repr_dim': 8,
            'require_pairwise_repr': False,

            'loss_factor': self.loss

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
                    'epochs': 20,
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
            'logging_per_update':1
        }

        self.checkpoint= {
            'save_path':'/home/jorey/pdhs/save/R/',
            'R_per_weight': './save/R/Repacking.pth',
            'J_per_weight': './save/J/FBB.pth',
            'G_per_weight':'/home/jorey/pdhs/save/G/onlyG_0.0_50.pth',
            'resume_G':False,



        }
        # R_Repacker_128_6_G256_fape_error_1__1_3   S40 Train without vio, and with fape
        script_path = os.path.dirname(os.path.abspath(__file__))



        self.output_dir=script_path+self.output_dir
        logging.info('Now creating new output dir on '+ self.output_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(  self.output_dir)




    def load_module(self,use_J):
        # build G model
        Rmodel = Repacker(**self.R_param,Str_encoder_param=self.str_encoder_param)
        self.Repacker = Rmodel.cuda()

        if use_J:
            logging.info("Using the fixbb method, the outputs will have new sequences")
            Jmodel = Joint_module(**self.J_param,G_param=self.G_param)
            self.Joint = Jmodel.cuda()

            if self.checkpoint['J_per_weight']:
                Jcheckpoint = torch.load(self.checkpoint['J_per_weight'], map_location=self.device)
                checkpoint_model = Jcheckpoint["base_model"]
                model_dict = self.Joint.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint_model.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.Joint.load_state_dict(checkpoint_model)
                del model_dict
                logging.info("load Joint model checkpoints")

            else:
                logging.info("not load model checkpoints")
        else:
            logging.info("Do not using the fixbb methods, the outputs use native or specified sequences  ")

        # load checkpoints
        if self.checkpoint['R_per_weight'] is not None:
            checkpoint_path=self.checkpoint['R_per_weight']
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            checkpoint_model = checkpoint["base_model"]
            model_dict = self.Repacker.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint_model.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.Repacker.load_state_dict(checkpoint_model)
            del model_dict
            logging.info("load Repacker model checkpoints")
        else:
            logging.info("not load model checkpoints")



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

    def use_J_generate_new_sequence(self,input):

        gtbatchs = {'gt_aatype': input['S'], 'gtframes_mask': input['mask']}
        result = self.Joint.design(input,gtbatchs,  **self.J_param)

        aatypes=result['aatype']
        return aatypes,result['recovery']


    def use_G_generate_new_sequence(self,input):

        gtbatchs = {'gt_aatype': input['S'], 'gtframes_mask': input['mask']}
        result = self.Joint.design_G(input,gtbatchs,  **self.J_param)


        # result = self.G.design(**input,  **self.G_param)

        aatypes=result['aatype']
        return aatypes,result['recovery']


    def design_step(self, epoch,name,use_J,use_amber):

        with torch.no_grad():
            # tools
            acc = AverageMeter(
                ['Ca_lddt', 'angles_acc1', 'angles_acc2', 'angles_acc3', 'angles_acc4',
                 'angles_acc5', 'angles_acc6', 'angles_acc7','recovery'])
            losses = AverageMeter(['bfloss', 'ca loss', 'chi loss', 'bb angles loss'])
            logging.info(f"\n[VAL] Start val epoch {epoch}", )
            t = tqdm.tqdm(self.testloader)
            val_step=0
            # test
            for proteins in t:
                if name!=None:
                    if proteins['domain_name'][0].decode('UTF-8') != name:
                        continue
                print('designing:',proteins['domain_name'][0].decode('UTF-8'))
                proteins=[proteins]
                x, aatypes, seq_masks, residue_indexs, gt_batchs = tied_features(
                    proteins, self.device, False if self.trainmodel == 'G' else True)

                input = {'X': x, 'S': aatypes, 'mask': seq_masks, 'residue_idx': residue_indexs}

                if use_J:
                    #      use_fbb_generate_new_sequence
                    aatypes,recovery=self.use_J_generate_new_sequence(input)
                    input['S'] = aatypes



                loss, result = self.Repacker.design(input, gt_batchs)


                # #update aatypes in proteins
                protein=self.make_new_protein(aatypes[0].detach().cpu().numpy().astype(int),result,proteins[0])

                if  use_J:
                    self.write_fa(proteins[0]['domain_name'][0].decode('UTF-8'),aatypes,proteins[0]['residue_index'],recovery)
                    acc.update(
                        [result['ca_lddt'].item(), result['Angle_accs'][0].item(), result['Angle_accs'][1].item(),
                         result['Angle_accs'][2].item()
                            , result['Angle_accs'][3].item(), result['Angle_accs'][4].item(),
                         result['Angle_accs'][5].item(),
                         result['Angle_accs'][6].item(),recovery])
                    losses.update([result['bfloss'].item(), result['fapeloss'].item(), result['chi_loss'].item(),
                                   result['bb_angle_loss'].item()])

                    if val_step % self.vis['logging_per_update'] == 0:
                        logging.info(
                            "ca_lddt:%.2f;Angle_accs1:%.2f;2:%.2f;3:%.2f;chi1:%.2f;chi2:%.2f;chi3:%.2f;chi4:%.2f;recovery:%.2f" % (
                                float(acc.avg(0)), float(acc.avg(1)), float(acc.avg(2)), float(acc.avg(3)),
                                float(acc.avg(4)), float(acc.avg(5)), float(acc.avg(6)), float(acc.avg(7)),
                                float(acc.avg(8))
                            ))
                        logging.info(
                            "bfloss:%.2f;fapeloss:%.2f;chi_angle_loss:%.2f;bb_angle_loss:%.2f;step %d；" % (
                                float(losses.avg(0)), float(losses.avg(1)), float(losses.avg(2)), float(losses.avg(3)),
                                self.global_step,
                            ))
                        logging.info(50 * '-')
                else:
                    acc.update(
                        [result['ca_lddt'].item(), result['Angle_accs'][0].item(), result['Angle_accs'][1].item(),
                         result['Angle_accs'][2].item()
                            , result['Angle_accs'][3].item(), result['Angle_accs'][4].item(),
                         result['Angle_accs'][5].item(),
                         result['Angle_accs'][6].item()])
                    losses.update([result['bfloss'].item(), result['fapeloss'].item(), result['chi_loss'].item(),
                                   result['bb_angle_loss'].item()])

                    if val_step % self.vis['logging_per_update'] == 0:
                        logging.info(
                            "ca_lddt:%.2f;Angle_accs1:%.2f;2:%.2f;3:%.2f;chi1:%.2f;chi2:%.2f;chi3:%.2f;chi4:%.2f;" % (
                                float(acc.avg(0)), float(acc.avg(1)), float(acc.avg(2)), float(acc.avg(3)),
                                float(acc.avg(4)), float(acc.avg(5)), float(acc.avg(6)), float(acc.avg(7))
                        ))
                        logging.info(
                            "bfloss:%.2f;fapeloss:%.2f;chi_angle_loss:%.2f;bb_angle_loss:%.2f;step %d；" % (
                                float(losses.avg(0)), float(losses.avg(1)), float(losses.avg(2)),
                                float(losses.avg(3)),
                                self.global_step,
                            ))
                        logging.info(50 * '-')

                self.relax(protein,use_amber)




                val_step=val_step+1

    def design_step_byseq(self, epoch,pre_name,pre_aatype):

        with torch.no_grad():
            # tools
            self.global_step=0
            acc = AverageMeter(
                ['Ca_lddt', 'angles_acc1', 'angles_acc2', 'angles_acc3', 'angles_acc4',
                 'angles_acc5', 'angles_acc6', 'angles_acc7','Recovery'])
            losses = AverageMeter(['q loss', 'ca loss', 'chi loss', 'bb angles loss'])
            logging.info(f"\n[VAL] Start val epoch {epoch}", )
            t = tqdm.tqdm(self.testloader)
            val_step=0
            # test
            for proteins in t:

                if pre_name!=None:

                    if proteins['domain_name'][0].decode('UTF-8')!=pre_name :
                        continue
                print('designing:',proteins['domain_name'][0].decode('UTF-8'))
                proteins=[proteins]
                x, aatypes, seq_masks, residue_indexs, gt_batchs = tied_features(
                    proteins, self.device, False if self.trainmodel == 'G' else True)

                if pre_aatype!=None:
                    index=proteins[0]['residue_index']-proteins[0]['residue_index'][0]
                    pre_aatype=pre_aatype[0][index]
                    aatypes=pre_aatype.unsqueeze(0)

                    # aatypes = pre_aatype
                    recovery=np.asarray([37])


                input = {'X': x, 'S': aatypes, 'mask': seq_masks, 'residue_idx': residue_indexs}
                input['S'] = aatypes
                loss, result = self.Repacker.design(input, gt_batchs,)


                # #update aatypes in proteins
                protein=self.make_new_protein(aatypes[0].detach().cpu().numpy().astype(int),result,proteins[0])


                self.relax(protein)
                #self.write_chis(**result,name=proteins[0]['domain_name'][0].decode('UTF-8'))

                acc.update([result['ca_lddt'].item(), result['Angle_accs'][0].item(), result['Angle_accs'][1].item(),
                            result['Angle_accs'][2].item()
                               , result['Angle_accs'][3].item(), result['Angle_accs'][4].item(),
                            result['Angle_accs'][5].item(),
                            result['Angle_accs'][6].item(), recovery.item()])
                losses.update([result['bfloss'].item(), result['ca_loss'].item(), result['chi_loss'].item(),
                               result['bb_angle_loss'].item()])

                if val_step % self.vis['logging_per_update'] == 0:
                    logging.info(
                        "quanteron_loss:%.2f;ca_loss:%.2f;chi_angle_loss:%.2f;bb_angle_loss:%.2f;step %d；" % (
                            float(losses.avg(0)), float(losses.avg(1)), float(losses.avg(2)), float(losses.avg(3)),
                            self.global_step,
                        ))

                    logging.info(
                        "ca_lddt:%.2f;Angle_accs1:%.2f;2:%.2f;3:%.2f;chi1:%.2f;chi2:%.2f;chi3:%.2f;chi4:%.2f;Recovery:%.2f;" % (
                            float(acc.avg(0)), float(acc.avg(1)), float(acc.avg(2)), float(acc.avg(3)),
                            float(acc.avg(4)), float(acc.avg(5)), float(acc.avg(6)), float(acc.avg(7)),
                            float(acc.avg(8)))
                    )
                val_step = val_step + 1


    def write_chis(self,chis,name,**kwargs):
        chis=chis.tolist()
        with open('/home/jorey/pdhs/designPDB/relax/chi_'+name+'.txt','w') as f:
            f.writelines(json.dumps(chis))

    def write_fa(self,name,aatype,res,recovery,add='A',notadd=True):
        aatype_list=aatype.detach().cpu().numpy().squeeze().tolist()
        def addX():
            seqs=[]
            start=res[0]
            for i in res:
                while start!=i:
                    seqs.append(add)
                    start = start + 1
                index=aatype_list.pop(0)
                seqs.append(restypes_with_x[index])
                start = start + 1


            assert len(aatype_list)==0
            return seqs
        if notadd:
            seqs=[]
            for i in aatype_list:
                seqs.append(restypes_with_x[i])
        else:
            seqs= addX()

        if not os.path.exists(       self.output_dir+'/fasta/'):
            os.mkdir(  self.output_dir+'/fasta/')
        seqs=''.join(i for i in seqs)



        with open(self.output_dir+'/fasta/'+name+'.fasta','w') as f:
            f.writelines('>'+name+' Recovery='+format(float(recovery),'.2f')+'\n')
            f.writelines(seqs+ '\n')
            f.close()

    def make_new_protein(self,aatype,result,oldprotein):

        import model.np.residue_constants as rc
        from data.data_transform import make_new_atom14_resid

        # change atoms mask to idea one
        ideal_atom_mask=rc.STANDARD_ATOM_MASK[aatype]
        finial_atom_mask=ideal_atom_mask#*proteins[0]['seqmask'][:,None]

        residx_atom37_to_atom14=make_new_atom14_resid(aatype).cpu().numpy()
        new={
            'aatype':aatype,
            'atom37_atom_exists': finial_atom_mask,
            'residx_atom37_to_atom14': residx_atom37_to_atom14,
            'final_14_positions': result['final_atom_positions'],
            'final_atom_mask': finial_atom_mask,
            'residue_index':oldprotein['residue_index'],
            'domain_name':oldprotein['domain_name'],
            'pred_b_factors':result['pred_b_factors']

        }

        return new
    def relax_one(self,files,name,output_name):
        from model.np import protein
        from model.network.relax_protein import relax_protein
        with open(files, 'r') as f:
            pdb_str = f.read()

        output_directory=self.output_dir+name
        design_protein=protein.from_pdb_string(pdb_str)
        logging.info(f"Running relaxation on ...")
        relax_protein('CUDA', design_protein, output_directory, output_name)



    def relax(self,proteins,use_amber=False):
        from model.network.feats import atom14_to_atom37
        from model.np import protein
        from model.network.relax_protein import relax_protein

        atom37=atom14_to_atom37(proteins['final_14_positions'].squeeze(0),proteins)
        pb_atom37=atom14_to_atom37(torch.as_tensor(proteins['pred_b_factors'][...,None]),proteins)
        #ADD UNK atoms
        #atom37=atom37+(1-proteins[0]['atom37_atom_exists'][:,:,None])*proteins[0]['all_atom_positions']


        result={'final_atom_positions': atom37,
             'final_atom_mask': proteins['final_atom_mask']}

        chain_index=np.zeros_like(proteins['aatype'],dtype=int)


        design_protein=protein.from_prediction(
            b_factors=pb_atom37.squeeze(-1),
            features=proteins,
            result=result,
            chain_index=chain_index
        )


        design_protein_str=protein.to_pdb(design_protein)
        unrelaxed_output_path = self.output_dir + '//PDHS_' + proteins['domain_name'][0].decode('UTF-8')


        with open(unrelaxed_output_path+'_unrelaxed.pdb', 'w') as fp:
            fp.write(design_protein_str)

        if use_amber:
            output_directory=self.output_dir+'//'
            output_name=proteins['domain_name'][0].decode('UTF-8')

            unrelaxed_output_path=unrelaxed_output_path+'_unrelaxed.pdb'
            logging.info(f"Running relaxation on {unrelaxed_output_path}...")
            relax_protein('CUDA', design_protein, output_directory, output_name)

            relaxname=output_directory+output_name+'_relaxed.pdb'
            logging.info(f"designed Relaxed output written to {relaxname}...")
    def design(self,name,use_J,use_amber):

        #load MODEL
        self.load_module(use_J) ##self.checkpoint['R_per_weight']
        # build dataset
        testloader = StructureDataset(self.dataset['designset'], **self.dataset)


        self.testloader = testloader

        self.global_step = self.R_scheduler['start_epoch'] * len(self.testloader)
        if use_J:
            self.Joint.zero_grad()
            self.Joint.eval()
        self.Repacker.zero_grad()
        self.Repacker.eval()

        self.design_step(0,name,use_J,use_amber)





    def design_byseq(self,name,seq):

        aatype=[restype_order[i] for i in seq]
        aatype=torch.as_tensor(aatype).cuda().unsqueeze(0)

        #load MODEL
        self.load_module() ##self.checkpoint['R_per_weight']
        # build dataset
        testloader = StructureDataset(self.dataset['designset'], **self.dataset)


        self.testloader = testloader

        self.global_step = self.R_scheduler['start_epoch'] * len(self.testloader)
        self.Joint.zero_grad()
        self.Joint.eval()
        self.Repacker.zero_grad()
        self.Repacker.eval()

        self.design_step_byseq(0,name,aatype)


    def dsign_formpnn(self):
        MPNNDIR='///home/jorey/ProDCoNN/casp14/'
        pdbs=glob.glob(MPNNDIR+'*.fasta')
        self.output_dir='//home/jorey/ProDCoNN/casp14pdb/'
        #load MODEL
        self.load_module() ##self.checkpoint['R_per_weight']
        # build dataset
        testloader = StructureDataset(self.dataset['designset'], **self.dataset)

        self.Joint.zero_grad()
        self.Joint.eval()
        self.Repacker.zero_grad()
        self.Repacker.eval()


        self.testloader = testloader

        for i in pdbs:
            name=i.split('/')[-1].split('.')[0].split('_')[0]

            if os.path.exists(args.output_dir+name+'_relaxed.pdb'):
                print('skip:',name)
                continue

            # if name!='T1145-D2':
            #     continue
            F=open(i,'r')
            F=F.readlines()
            seq=F[-1].split('\n')[0].replace('X','')


            aatype = [restype_order_with_x[i] for i in seq]
            aatype = torch.as_tensor(aatype).cuda().unsqueeze(0)

            #try:
            self.design_step_byseq(0, name.upper(), aatype)
            # except:
            #     print('error:',name)


def read_fasta(file_path):
    protein_names = []
    protein_sequences = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        name = ""
        sequence = ""

        for line in lines:
            line = line.strip()

            if line.startswith('>'):
                # Process the previous protein entry
                if name and sequence:
                    protein_names.append(name)
                    protein_sequences.append(sequence)

                # Start a new protein entry
                name = line[1:]
                sequence = ""
            else:
                sequence += line

        # Process the last protein entry
        if name and sequence:
            protein_names.append(name)
            protein_sequences.append(sequence)

    return protein_names, protein_sequences




if __name__ == "__main__":
    # read_fasta()
    args=argsx()
    args.just_valastrain=True
    args.dataset['shuffle']=False
    args.R_scheduler['scheduler']['kwargs']['epochs']=1
    args.vis['logging_per_update']=1


    # Create an argument parser
    parser = argparse.ArgumentParser(description='Process PDB input and processed files output  directories')

    # Add arguments for input and output directories
    parser.add_argument('--use_amber', action='store_true', default=False, help='Using Amber for optimizing the output results')
    parser.add_argument('--use_fbb', action='store_true', default=False, help='Using fbb to generate new sequences')


    # Parse the command-line arguments
    argss = parser.parse_args()
    print('--use_amber=',argss.use_amber)
    print('--use_fbb=',argss.use_fbb)
    # Call the function with the provided input and output directories
    args.design(None,argss.use_fbb,argss.use_amber)
