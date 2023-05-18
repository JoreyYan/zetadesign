import torch
import torch.nn as nn
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
from CLIP.data import builder_for_AL

import tqdm
import time
from CLIP.model.utils import AverageMeter
from CLIP.model.metrics import Metrics
from sklearn.metrics import confusion_matrix
import math
from CLIP.model.network.Joint_model import Joint_module
from set_logger import set_logger1
import logging
set_logger1()
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
import json
import matplotlib.pyplot as pyplot




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
def setup_seed(seed=20000):
    import random
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
# setup_seed()
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

def _S_to_seq(S, mask):
    alphabet = 'XACDEFGHIKLMNPQRSTVWY'
    # seq = ''.join([alphabet[c] for c, m in zip(S.tolist(), mask.tolist()) if m > 0])

    seq = ''.join([alphabet[c] for c, m in zip(S.tolist(), mask.tolist()) ])
    return seq


#[!!!] actually 30 is missing_mask, 1 is padding mask

class argsx():
    def __init__(self):
        self.resume =True
        self.muliti_GPU=False

        self.experiment_path='/home/junyu/桌面/code/ProteinCLIP/CLIP/save/'
        self.config='Clip.yaml'

        self.val_freq=1
        self.data='Biochem_test'


        self.Generator_checkpoint_path = self.experiment_path + '6layersmeshmpnnsse_43bad_trans_noise0.1_shift67.pt'#'/6layersmeshmpnnsse_43bad_trans_noise001_chem02__shift54' + '.pt' 6layersmeshmpnnsse_43bad_trans_noise001_chem02__shift54   6layersmeshmpnnsse_43bad_trans_shift38   6layersmeshmpnnsse_43bad_trans_noise002_shift60
        self.CLIP_checkpoint_path = self.experiment_path + 'G_0.1k1_lamb_8msa__IPA_3recyle_5noise_Norm_chemNoloss_2-last.pth'#'T00_12msa__SE_2recyle01noise_2msalaryer_nochemloss1-last.pth'#'/T00_8msa__SE_2recyle02noise_msalaryer6_nochem0-last' + '.pth'  #12msa__SE_2recyleNoise22-last  #_chemf_T01_12msa__SE_3recyle02noise0-last
        # chemloss_chemf_T00_8msa__SE_2recyle02noise_msalaryer310-last 是
        # chemloss_chemf_T00_12msa__SE_2recyle02noise1 2层msa
        # zuinew T00_12msa__SE_2recyle002noise_2msalaryer_nochemloss0
        self.eval_mode=True
        self.msa=8

        self.backbone_noise=0.1
        self.Joint_Factor=10






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

    SEQGenerator = builder_for_AL.model_builder(args,config.model,model='SEQGenerator',backbone_noise=args.backbone_noise)
    Joint_OPT =Joint_module( n_module_str=4, n_layer=3,IPA_layer=2,)#builder_for_AL.model_builder(args,config.model,model='SEQCLIP_GPT') # #
    if args.muliti_GPU:
        SEQGenerator = nn.DataParallel(SEQGenerator)
    SEQGenerator=SEQGenerator.cuda()
    Joint_OPT=Joint_OPT.cuda()





    #load checkpoints
    if args.resume:
        checkpoint = torch.load(args.Generator_checkpoint_path, map_location=device)
        checkpoint_model=checkpoint["model_state_dict"]
        model_dict = SEQGenerator.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint_model.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        SEQGenerator.load_state_dict(model_dict)


        checkpoint = torch.load(args.CLIP_checkpoint_path, map_location=device)
        checkpoint_model=checkpoint["base_model"]
        # new_state={}
        # for k,v in checkpoint_model.items():
        #     new_state[k[7:]]=v
        CLIPmodel_dict = Joint_OPT.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint_model.items() if k in CLIPmodel_dict}
        CLIPmodel_dict.update(pretrained_dict)
        Joint_OPT.load_state_dict(checkpoint_model)


        del CLIPmodel_dict
        del model_dict



    #beginning to train
    start_epoch = 0
    logging.info("entring epochs")

    SEQGenerator.eval()

    Joint_OPT.zero_grad()
    # SEQCLIP.add_iter(1)
    SEQGenerator.zero_grad()

    # build dataset
    eval_mode = args.eval_mode
    if eval_mode:
        val_dataloader = builder_for_AL.dataset_builder(args, config, justval=eval_mode, justtest=False,
                                                          dataset=args.data)





    num_iter = 0
    n_batches = len(val_dataloader)
    idx=0
    epoch=0
    msa_num = args.msa

    with torch.no_grad():
        validate_trainCLIP(SEQGenerator, Joint_OPT, val_dataloader, epoch, config, ((1 + epoch) * n_batches), msa_num)



def _get_recovery(S, pred, mask):



    true = (S * mask).detach().type(torch.int)

    this_correct = ((pred == true).sum() - (1 - mask.detach()).sum())
    thisnods = torch.sum(mask)
    seq_recovery_rate = 100 * this_correct / thisnods



    return seq_recovery_rate


def validate_trainCLIP(SEQGenerator,Joint_OPT, test_dataloader, epoch,  config, n_itr,msanum=6):
    logging.info(f"\n[VALIDATION] Start CLIP validating epoch {epoch}", )
    Joint_OPT.eval()

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    acc = AverageMeter([ 'recovery_acc','recovery_acc'])
    t = tqdm.tqdm(test_dataloader)
    test_losses = AverageMeter(['CE loss','clip loss'])

    num_iter = 0
    confusion = np.zeros((21, 21))
    confusion_G = np.zeros((21, 21))
    if args.msa==1:
        use_SE=False
    else:
        use_SE = True


    idx = 0
    base_folder='/home/junyu'
    ali_file = base_folder + '/IPA_msafirst'+args.data+'nochemloss_Norm_chemloss_noise'+str(args.Joint_Factor)+str(args.backbone_noise)+'_msa'+str(args.msa)+'_last' + '.fa'



    with open(ali_file, 'wb') as f:
        f.write(args.Generator_checkpoint_path.encode())
        f.write('\n'.encode())
        f.write(args.CLIP_checkpoint_path.encode())
        f.write('\n'.encode())
        f.write(args.data.encode())
        f.write('\n'.encode())
        f.write('\n'.encode())
        f.write(str(args.msa).encode())
        f.write('\n'.encode())
        for proteins in t:
            proteins=[proteins]
            num_iter += 1
            n_itr = n_itr
            # pre_deal data; get points
            points, real_tokens, residue_idx, mask, seq_for_conveter,SSE3_seq,SSE8_seq = builder_for_AL.tied_featurize(proteins,device)  # B,L
            real_tokens[mask==0]=0  # pading and miss is 0

            # if points.shape[1]>180:
            #     continue

            pair=0
            faketokenlist=[]
            loss_CE_all=0
            recovery_acc_all=0

            for i in range(msanum):
                loss_CE,recovery_acc,fake_token,h_V,cm=SEQGenerator(points,real_tokens,mask,residue_idx,SSE3_seq,SSE8_seq,Index_embed=True,use_gradient=False)
                confusion_G=confusion_G+cm


                faketokenlist.append(fake_token)
                loss_CE_all=loss_CE_all+loss_CE
                recovery_acc_all=recovery_acc_all+recovery_acc
            seq_fake=fake_token[0]
            seq_fake=_S_to_seq(seq_fake,mask[0])


            faketokenlist=torch.stack(faketokenlist,dim=1)

            loss_CE=loss_CE_all/msanum
            recovery_acc=recovery_acc_all/msanum


            if use_SE:
                loss_CE_msa, recovery_acc_msa, dis,pred = Joint_OPT(faketokenlist, real_tokens, points, mask, residue_idx,
                                                               args.Joint_Factor,inference=True)
                seq_msa = _S_to_seq(pred[0], torch.ones_like(mask[0]))
                f.write(
                    '>name={}, Energyof G={}, Energyof MSA={}, seq_recovery_G={}, seq_recovery_MSA={}\nSeq_Origin={}\nSeq_G={}\nSeq_MSA={}\n\n\n'.format(
                        proteins[0]['name'], np.format_float_positional(loss_CE, unique=False, precision=4),
                        np.format_float_positional(loss_CE_msa, unique=False, precision=4),
                        np.format_float_positional(recovery_acc, unique=False, precision=4),
                        np.format_float_positional(recovery_acc_msa, unique=False, precision=4),
                        seq_for_conveter[0][0][1], seq_fake, seq_msa).encode())
            else:
                loss_CE_msa=0*loss_CE
                recovery_acc_msa = 0 * recovery_acc
                dis=0
                f.write(
                    '>name={}, Energyof G={},  seq_recovery_G={}, Seq_Origin={}\nSeq_G={}\n\n\n\n'.format(
                        proteins[0]['name'], np.format_float_positional(loss_CE, unique=False, precision=4),

                        np.format_float_positional(recovery_acc, unique=False, precision=4),

                        seq_for_conveter[0][0][1], faketokenlist, ).encode())

            confusion=confusion+cm



            CLIP_loss=loss_CE_msa+1*dis
            # update records
            test_losses.update([loss_CE.item(), CLIP_loss.item()])
            acc.update([recovery_acc.item(),recovery_acc_msa.item()])


            idx = idx + 1



            t.set_description(
                "CEloss:%.2f;msa loss:%.2f;recovery:%.3f;recoverymsa:%.3f;dis %f；step %d；" % (
                    float(test_losses.avg(0)), float(test_losses.avg(1)), float(acc.avg(0)),
                    float(acc.avg(1)), float(dis), n_itr,
                    ))

    f.close()
    # if use_SE:
    #     np.save('chem_MSA_add2_comfu.npy',confusion) #tem=0
    # else:
    #     np.save('chem_G_comfu.npy',confusion_G)

    # lookup='XACDEFGHIKLMNPQRSTVWY'
    #
    # plot_confusion_matrix(confusion,list(lookup))
    # print_confusion(confusion,lookup)
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


def print_confusion(mat, lookup):
    counts = mat.astype(np.int32)
    mat = (counts.T / counts.sum(axis=-1, keepdims=True).T).T

    mat = np.round(mat * 1000).astype(np.int32)
    # mat = mat.astype(np.int32)
    res = '\n'
    for i in range(21):
        res += '\t{}'.format(lookup[i])
    res += '\tCount\n'
    for i in range(21):
        res += '{}\t'.format(lookup[i])
        res += '\t'.join('{}'.format(n) for n in mat[i])
        res += '\t{}\n'.format(sum(counts[i]))
    print(res)
import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot
    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    title:        the text to display at the top of the matrix
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


if __name__ == "__main__":
    args=argsx()
    config = builder_for_AL.cfg_from_yaml_file(args.config )
    config.data.max_length=600


    ####init Generator
    # run_Generatornet(args,config)

    ####alter train
    #
    # logging.info(f"------------------------run the th CLIP -----------------------")
    train_CLIPnet(args,config)
    # logging.info(f"------------------------run the {i} th train_Generatorby_CLIP -----------------------")
    # train_Generatorby_CLIP(args,config)

    # x=[]
    # file='/home/junyu/下载//B6I6P1/B6I6P1_61f22.a3m'
    # with open(file) as f:
    #     c=f.readlines()
    #     for i in range(len(c)):
    #         if c[i][0]=='>':
    #             x.append(c[i+1])
    #
    #
    # xxx='XACDEFGHIKLMNPQRSTVWY-'
    # SEQ_FAKE_LIST=[]
    # DIST=torch.zeros((160,22))
    # faketokenlist=[]
    # for i in x:
    #     i=i.split('\n')[0]
    #     for a in i:
    #         try :
    #             vv=xxx.index(a.upper())
    #         except:
    #             print(a)
    #     indices = np.asarray([xxx.index(a.upper()) for a in i], dtype=np.int32)
    #     faketokenlist.append(indices)
    #     # if len(faketokenlist)>20:
    #     #     break
    #
    # for i in faketokenlist:
    #
    #     for j in range(160):
    #         DIST[j, int(i[j])] = DIST[j, int(i[j])] + 1
    #
    # DIST=DIST[:,:-1]
    # DIST = np.asarray(DIST.transpose(0, 1))
    # pyplot.matshow(DIST)
    # pyplot.show()



