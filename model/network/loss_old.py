
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import ml_collections

from model.np import residue_constants

from ..Rigid import Rigid,Rotation

def compute_fape(
    pred_frames: Rigid,
    target_frames: Rigid,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    l1_clamp_distance: Optional[float] = None,
    eps=1e-8,
) -> torch.Tensor:
    """
        Computes FAPE loss.
        Args:
            pred_frames:
                [*, N_frames] Rigid object of predicted frames
            target_frames:
                [*, N_frames] Rigid object of ground truth frames
            frames_mask:
                [*, N_frames] binary mask for the frames
            pred_positions:
                [*, N_pts, 3] predicted atom positions
            target_positions:
                [*, N_pts, 3] ground truth positions
            positions_mask:
                [*, N_pts] positions mask
            length_scale:
                Length scale by which the loss is divided
            l1_clamp_distance:
                Cutoff above which distance errors are disregarded
            eps:
                Small value used to regularize denominators
        Returns:
            [*] loss tensor
    """
    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]

    # FP16-friendly averaging. Roughly equivalent to:
    #
    # norm_factor = (
    #     torch.sum(frames_mask, dim=-1) *
    #     torch.sum(positions_mask, dim=-1)
    # )
    # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
    #
    # ("roughly" because eps is necessarily duplicated in the latter)
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = (
        normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
    )
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    return normed_error


def backbone_loss(
    backbone_rigid_tensor: torch.Tensor,
    backbone_rigid_mask: torch.Tensor,
    traj: torch.Tensor,
    use_clamped_fape: Optional[torch.Tensor] = None,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:
    pred_aff = Rigid.from_tensor_7(traj)
    pred_aff = Rigid(
        Rotation(rot_mats=pred_aff.get_rots().get_rot_mats(), quats=None),
        pred_aff.get_trans(),
    )

    # DISCREPANCY: DeepMind somehow gets a hold of a tensor_7 version of
    # backbone tensor, normalizes it, and then turns it back to a rotation
    # matrix. To avoid a potentially numerically unstable rotation matrix
    # to quaternion conversion, we just use the original rotation matrix
    # outright. This one hasn't been composed a bunch of times, though, so
    # it might be fine.
    gt_aff = Rigid.from_tensor_4x4(backbone_rigid_tensor)

    fape_loss = compute_fape(
        pred_aff,
        gt_aff[None],
        backbone_rigid_mask[None],
        pred_aff.get_trans(),
        gt_aff[None].get_trans(),
        backbone_rigid_mask[None],
        l1_clamp_distance=clamp_distance,
        length_scale=loss_unit_distance,
        eps=eps,
    )
    if use_clamped_fape is not None:
        unclamped_fape_loss = compute_fape(
            pred_aff,
            gt_aff[None],
            backbone_rigid_mask[None],
            pred_aff.get_trans(),
            gt_aff[None].get_trans(),
            backbone_rigid_mask[None],
            l1_clamp_distance=None,
            length_scale=loss_unit_distance,
            eps=eps,
        )

        fape_loss = fape_loss * use_clamped_fape + unclamped_fape_loss * (
            1 - use_clamped_fape
        )

    # Average over the batch dimension
    fape_loss = torch.mean(fape_loss)

    return fape_loss


def fape_loss(
        out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        config: ml_collections.ConfigDict,
) -> torch.Tensor:
    bb_loss = backbone_loss(
        traj=out["sm"]["frames"],
        **{**batch, **config.backbone},
    )

    sc_loss = sidechain_loss(
        out["sm"]["sidechain_frames"],
        out["sm"]["positions"],
        **{**batch, **config.sidechain},
    )

    loss = config.backbone.weight * bb_loss + config.sidechain.weight * sc_loss

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss



def masked_mean(mask, value, dim, eps=1e-4):
    mask = mask.expand(*value.shape)
    # C=(mask * value).detach().cpu().numpy()
    #D=torch.sum(mask * value, dim=(0,1))/ (eps + torch.sum(mask, dim=(0,1)))
    # print(D)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))

def supervised_chi_loss(
        angles_sin_cos: torch.Tensor,
        unnormalized_angles_sin_cos: torch.Tensor,
        aatype: torch.Tensor,
        seq_mask: torch.Tensor,
        chi_mask: torch.Tensor,
        gt_angles_sin_cos: torch.Tensor,
        chi_weight: float,
        angle_norm_weight: float,
        eps=1e-6,
        **kwargs,
) -> torch.Tensor:
    """
        Implements Algorithm 27 (torsionAngleLoss)
        Args:
            angles_sin_cos:
                [*, N, 7, 2] predicted angles
            unnormalized_angles_sin_cos:
                The same angles, but unnormalized
            aatype:
                [*, N] residue indices
            seq_mask:
                [*, N] sequence Angle mask
            chi_mask:
                [*, N, 4] angle mask
            chi_angles_sin_cos:
                [*, N, 7, 2] ground truth angles
            chi_weight:
                Weight for the angle component of the loss
            angle_norm_weight:
                Weight for the normalization component of the loss
        Returns:
            [*] loss tensor
    """
    pred_angles = angles_sin_cos[..., 3:, :]
    residue_type_one_hot = torch.nn.functional.one_hot(
        aatype,
        residue_constants.restype_num + 1,
    )
    chi_pi_periodic = torch.einsum(
        "...hij,jk->hik",
        residue_type_one_hot.type(angles_sin_cos.dtype),
        angles_sin_cos.new_tensor(residue_constants.chi_pi_periodic),
    )

    true_chi = gt_angles_sin_cos[..., 3:, :]

    shifted_mask = (1 - 2 * chi_pi_periodic).unsqueeze(-1)
    true_chi_shifted = shifted_mask * true_chi
    sq_chi_error = torch.sum((true_chi - pred_angles) ** 2, dim=-1)
    sq_chi_error_shifted = torch.sum(
        (true_chi_shifted - pred_angles) ** 2, dim=-1
    )
    sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)



    sq_chi_loss = masked_mean(
        chi_mask[...,  :, :], sq_chi_error, dim=(-1,-2,-3 )
    )


    # compute in degree
    angles_diffs=torch.sum(chi_mask * sq_chi_error, dim=(0, 1)) / (1e-4 + torch.sum(chi_mask, dim=(0, 1)))
    angles_diffs=180*torch.arccos((2-angles_diffs)/2)/torch.pi


    loss = chi_weight * sq_chi_loss#+chi_weight*angles_diff*torch.pi/180

    angle_norm = torch.sqrt(
        torch.sum(unnormalized_angles_sin_cos ** 2, dim=-1) + eps
    )
    norm_error = torch.abs(angle_norm - 1.0)
    norm_error = norm_error.permute(
        *range(len(norm_error.shape))[1:-2], 0, -2, -1
    )
    angle_norm_loss = masked_mean(
        seq_mask[...,  :, None], norm_error, dim=(-1, -2)
    )

    loss = loss + angle_norm_weight * angle_norm_loss

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss,angles_diffs

def bb_torsion_loss(
        angles_sin_cos: torch.Tensor,
        seq_mask: torch.Tensor,
        bb_angle_mask: torch.Tensor,
        gt_angles_sin_cos: torch.Tensor,
        bb_torsion_weight: float,

        eps=1e-6,
        **kwargs,
) -> torch.Tensor:
    """
        Implements Algorithm 27 (torsionAngleLoss)
        Args:
            angles_sin_cos:
                [*, N, 7, 2] predicted angles


            seq_mask:
                [*, N] sequence mask

            chi_angles_sin_cos:
                [*, N, 7, 2] ground truth angles
            chi_weight:
                Weight for the angle component of the loss
            angle_norm_weight:
                Weight for the normalization component of the loss
        Returns:
            [*] loss tensor
    """
    pred_angles = angles_sin_cos[..., :3, :]

    true_angles = gt_angles_sin_cos[..., :3, :]


    sq_chi_error = torch.sum((true_angles - pred_angles) ** 2, dim=-1)
    #A=bb_angle_mask.detach().cpu().numpy()
    #sq_chi_error=bb_angle_mask*sq_chi_error
    #B=sq_chi_error.detach().cpu().numpy()
    sq_chi_loss = masked_mean(
        bb_angle_mask[...,  :], sq_chi_error, dim=(-1,-2,-3 )
    )
    # C=sq_chi_loss.detach().cpu().numpy()



    # angle in degrees
    angles_diff = torch.sum(bb_angle_mask * sq_chi_error, dim=(0, 1)) / (1e-4 + torch.sum(bb_angle_mask, dim=(0, 1)))
    angles_diffs=180*torch.arccos((2-angles_diff)/2)/torch.pi



    loss = bb_torsion_weight * sq_chi_loss#+bb_torsion_weight*angles_diff

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss,angles_diffs



def Frame_Aligned_error( pred_frames: Rigid,
    gt_frames: Rigid,
    gt_frames_mask: torch.Tensor,

    eps=1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''

    :param pred_frames: [*, N_frames] Rigid object of predicted frames , we use N_frames=1
    :param target_frames: [*, N_frames] Rigid object of ground truth frames, we use N_frames=1
    :param frames_mask:


    :param eps:
    :return:
    '''
    #pred_frames=Rigid.from_tensor_4x4(pred_frames).to_tensor_7()
    gt_frames=Rigid.from_tensor_4x4(gt_frames).to_tensor_7()

    # xloss=torch.mean(torch.sqrt(
    #     torch.sum((pred_frames - gt_frames) ** 2, dim=-1) + eps
    # ))


    pred_quanteron=pred_frames[...,:,:,:4]
    gt_quanteron=gt_frames[...,:,:,:4]

    error_q_dist = torch.sqrt(
        torch.sum((pred_quanteron - gt_quanteron) ** 2, dim=-1) + eps
    )
    normed_q_error = error_q_dist* gt_frames_mask
    normed_q_error=torch.sum(normed_q_error,-1)

    normed_q_error = (
        normed_q_error / (eps + torch.sum(gt_frames_mask, dim=-1))#[..., None]
    )
    #norm in batch dims
    normed_q_error =torch.mean(normed_q_error)

    pred_trans=pred_frames[...,:,:,4:]
    gt_trans=gt_frames[...,:,:,4:]
    error_t_dist = torch.sqrt(
        torch.sum((pred_trans - gt_trans) ** 2, dim=-1) + eps
    )
    normed_t_error =error_t_dist* gt_frames_mask
    normed_t_error=torch.sum(normed_t_error,-1)
    normed_t_error = (
        normed_t_error / (eps + torch.sum(gt_frames_mask, dim=-1))#[..., None]
    )
    normed_t_error = torch.mean( normed_t_error )
    # normed_t_error=torch.as_tensor(0)
    # normed_q_error=torch.as_tensor(0)
    # normed_q_error=xloss
    # normed_t_error=xloss
    assert not torch.any(torch.isnan(normed_t_error))
    return normed_q_error, normed_t_error

def new_chi_mask(aatype):
    '''

    :param aatype:  B,L, 0~19 list ofsequences
    :return: B,L,7 angle mask accoding to the aa
    '''
    chi_angles_mask = list(residue_constants.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = torch.tensor(chi_angles_mask)

    chis_mask = chi_angles_mask[aatype, :]

    return chis_mask

def torsion_angle_loss(
        angles_sin_cos: torch.Tensor,
        unnormalized_angles_sin_cos: torch.Tensor,
        aatype: torch.Tensor,
        gtframes_mask: torch.Tensor,
        gt_angles_mask: torch.Tensor,
        gt_angles_sin_cos: torch.Tensor,
        chi_weight: float,
        bb_torsion_weight:float,
        angle_norm_weight: float,
        eps=1e-6,
        requires_degree=False,
        just_bb=False,
        **kwargs,
) -> torch.Tensor:

    if not just_bb:


        chi_mask=gt_angles_mask[:,:,3:]

        # error=chi_mask!=newchimask
        # x=torch.where(chi_mask!=newchimask)
        # for i,j,k in zip(x[0],x[1],x[2]):
        #     print(chi_mask[i,j,k])
        # gts=chi_mask[7].detach().cpu().numpy()
        # ncs=newchimask[7].detach().cpu().numpy()
        # print(torch.sum(error))

        seq_mask=gtframes_mask[:,:,0]


        chi_loss,angle_diff_chi=supervised_chi_loss(
            angles_sin_cos,
            unnormalized_angles_sin_cos,
            aatype,
            seq_mask,
            chi_mask,
            gt_angles_sin_cos,
            chi_weight,
            angle_norm_weight,
            eps,
        ** kwargs,)


    bb_angle_mask=gt_angles_mask[:,:,:3]
    seq_angle_mask=bb_angle_mask[:,:,0]*bb_angle_mask[:,:,1]*bb_angle_mask[:,:,2]
    bb_angle_loss,angle_diff_bb=bb_torsion_loss(
        angles_sin_cos,
        seq_angle_mask,
        bb_angle_mask,
        gt_angles_sin_cos,
        bb_torsion_weight,
    )

    if just_bb:
        if requires_degree:
            return  bb_angle_loss, angle_diff_bb
        else:
            return  bb_angle_loss

    else:
        if requires_degree:
            return chi_loss, bb_angle_loss,angle_diff_bb,angle_diff_chi
        else:
            return chi_loss, bb_angle_loss


def aa_loss_smoothed(
        logits:torch.tensor,
        gt_aatype:torch.tensor,
        seq_mask:torch.tensor,
        aatempweight:float,
        kind:int,
        **kwargs,
):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(gt_aatype, kind).float()

    # Label smoothing
    S_onehot = S_onehot + aatempweight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    log_probs=torch.nn.functional.log_softmax(logits, dim=-1)
    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * seq_mask) / torch.sum(seq_mask)
    return loss, loss_av


def FBB_loss(out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
         fae_t_f: float,
         fae_q_f: float,
         aa_f: float,
            angle_f:float,**kwargs
) -> torch.Tensor:
    bb_fae_loss_q,bb_fae_loss_t = Frame_Aligned_error( pred_frames=out['sidechain_frames'][...,:,0,:],gt_frames=batch["gtframes"][...,:,0,:],gt_frames_mask=batch['gtframes_mask'])


    bb_angle_loss,angle_diff_bb = torsion_angle_loss(**out, **batch,requires_degree=True,just_bb=True,**kwargs )
    # mask X
    Xmask=batch['gt_aatype']!=20
    seq_mask=Xmask*batch['gtframes_mask']
    gt_aatype=batch['gt_aatype']*Xmask


    score, bb_loss_av=aa_loss_smoothed(
        gt_aatype=gt_aatype,
        logits=out['logits'],
        seq_mask=seq_mask,**kwargs,)



    loss = fae_q_f * bb_fae_loss_q+fae_t_f*bb_fae_loss_t + aa_f * bb_loss_av+angle_f* bb_angle_loss

    result={
        'quanteron_loss':bb_fae_loss_q.detach().cpu(),
        'ca_loss': bb_fae_loss_t.detach().cpu(),
        'bb_angle_loss':bb_angle_loss.detach().cpu(),
        'aa_loss':bb_loss_av.detach().cpu(),
    }

    gt_ca=batch['gtframes'].get_trans().detach()
    pred_ca=out['frames'][:,:,4:].detach()
    ca_lddt=lddt(pred_ca,gt_ca,true_points_mask=batch['gtframes_mask'].unsqueeze(-1))
    ca_lddt=torch.mean(ca_lddt)*100

    Angle_accs = angle_diff_bb.detach()

    result.update({
        'finial_loss': loss.detach().cpu(),
        'ca_lddt': ca_lddt.cpu(),
        'Angle_accs': Angle_accs.cpu(),
    })


    return loss,result





def Repackerloss(out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
         fae_t_f: float,
         fae_q_f: float,
            angle_f:float,**kwargs
) -> torch.Tensor:
    # xx=out["sidechain_frames"][...,:,0,:]-out["frames"]
    # yy=torch.mean(xx)
    bb_fae_loss_q,bb_fae_loss_t = Frame_Aligned_error( pred_frames=out["frames"],gt_frames=batch["gtframes"][...,:,0,:,:],gt_frames_mask=batch['gtframes_mask'][...,:,0])

    # framesloss=torch.sqrt(
    #     torch.sum((out["sidechain_frames"] - batch["gtframes"]) ** 2, dim=-1)
    # )
    # framesloss=torch.sum(framesloss,(0,1,-1))/(torch.sum(batch['gtframes_mask'],(0,1))+1e-4)
    # framesloss=torch.mean(framesloss)

    chi_loss,bb_angle_loss,angle_diff_bb,angle_diff_chi = torsion_angle_loss(**out, **batch,requires_degree=True,**kwargs )


    loss = fae_q_f * bb_fae_loss_q+fae_t_f*bb_fae_loss_t + angle_f * (chi_loss+bb_angle_loss)#+framesloss

    result={
        'quanteron_loss':bb_fae_loss_q.detach().cpu(),
        'ca_loss': bb_fae_loss_t.detach().cpu(),
        'chi_loss': chi_loss.detach().cpu(),
        'bb_angle_loss': bb_angle_loss.detach().cpu(),

    }

    gt_ca=Rigid.from_tensor_4x4(batch['gtframes'][...,:,0,:,:]).get_trans().detach()
    pred_ca=out['frames'][:,:,4:].detach()
    ca_lddt=lddt(pred_ca,gt_ca,true_points_mask=batch['gtframes_mask'][:,:,0].unsqueeze(-1))
    ca_lddt=torch.mean(ca_lddt)*100

    Angle_accs=torch.cat((angle_diff_bb,angle_diff_chi),dim=0).detach().type(torch.int)


    result.update({
        'finial_loss': loss.detach().cpu(),
        'ca_lddt': ca_lddt.cpu(),
        'Angle_accs': Angle_accs.cpu(),
    })


    return loss,result

def Repacker_Aux_loss(out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],

             fae_t_f: float,
             fae_q_f: float,
    angle_f:float,
   **kwargs) -> torch.Tensor:

    #bb_fae_loss_q,bb_fae_loss_t = Frame_Aligned_error( pred_frames=out['sidechain_frames'],gt_frames=batch["gtframes"],gt_frames_mask=batch['gtframes_mask'])
    bb_fae_loss_q=0
    bb_fae_loss_t=0
    chi_loss,bb_angle_loss = torsion_angle_loss(**out, **batch,**kwargs )


    loss = fae_q_f * bb_fae_loss_q+fae_t_f*bb_fae_loss_t + angle_f * (chi_loss+bb_angle_loss)



    return loss

def lddt(predicted_points,
         true_points,
         true_points_mask,
         cutoff=15.,
         per_residue=False):
  """Measure (approximate) lDDT for a batch of coordinates.
  lDDT reference:
  Mariani, V., Biasini, M., Barbato, A. & Schwede, T. lDDT: A local
  superposition-free score for comparing protein structures and models using
  distance difference tests. Bioinformatics 29, 2722–2728 (2013).
  lDDT is a measure of the difference between the true distance matrix and the
  distance matrix of the predicted points.  The difference is computed only on
  points closer than cutoff *in the true structure*.
  This function does not compute the exact lDDT value that the original paper
  describes because it does not include terms for physical feasibility
  (e.g. bond length violations). Therefore this is only an approximate
  lDDT score.
  Args:
    predicted_points: (batch, length, 3) array of predicted 3D points
    true_points: (batch, length, 3) array of true 3D points
    true_points_mask: (batch, length, 1) binary-valued float array.  This mask
      should be 1 for points that exist in the true points.
    cutoff: Maximum distance for a pair of points to be included
    per_residue: If true, return score for each residue.  Note that the overall
      lDDT is not exactly the mean of the per_residue lDDT's because some
      residues have more contacts than others.
  Returns:
    An (approximate, see above) lDDT score in the range 0-1.
  """


  assert len(predicted_points.shape) == 3
  assert predicted_points.shape[-1] == 3
  assert true_points_mask.shape[-1] == 1
  assert len(true_points_mask.shape) == 3

  # Compute true and predicted distance matrices.  两两之间的距离矩阵
  dmat_true = torch.sqrt(1e-10 + torch.sum((true_points[:, :, None] - true_points[:, None, :])**2, axis=-1))

  dmat_predicted = torch.sqrt(1e-10 + torch.sum(
      (predicted_points[:, :, None] -
       predicted_points[:, None, :])**2, axis=-1))


  #找出两两之间距离小于15的部分
  dists_to_score = (
      (dmat_true < cutoff).type(torch.float32) * true_points_mask *
      torch.transpose(true_points_mask, 2, 1) *
      (1. - torch.eye(dmat_true.shape[1]).cuda())  # Exclude self-interaction.
  )

  # Shift unscored distances to be far away.
  dist_l1 = torch.abs(dmat_true - dmat_predicted)

  # True lDDT uses a number of fixed bins. 都小于0.5的话，score会是1  都大于4 结果是0
  # We ignore the physical plausibility correction to lDDT, though.
  score = 0.25 * ((dist_l1 < 0.5).type(torch.float32) +
                  (dist_l1 < 1.0).type(torch.float32) +
                  (dist_l1 < 2.0).type(torch.float32) +
                  (dist_l1 < 4.0).type(torch.float32))

  # Normalize over the appropriate axes.
  reduce_axes = (-1,) if per_residue else (-2, -1)
  norm = 1. / (1e-10 + torch.sum(dists_to_score, axis=reduce_axes))
  score = norm * (1e-10 + torch.sum(dists_to_score * score, axis=reduce_axes))

  return score