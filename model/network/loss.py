
from functools import partial
import numpy as np
# import pandas
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import ml_collections as mlc
from model.network.repackerloss import find_structural_violations,violation_loss
from model.np import residue_constants
# from model.np.dunbrack.read import get_dunbrak,fake_energy
from ..Rigid import Rigid,Rotation

# dunbrak = get_dunbrak()
# dunbrak = torch.tensor(dunbrak.values, dtype=torch.float)



HuberLoss=nn.HuberLoss(reduction='none', delta=0.25)


eps=1e-4
loss_config = mlc.ConfigDict(
        {
            "loss": {
                "fape": {
                    "backbone": {
                        "clamp_distance": 10.0,
                        "loss_unit_distance": 10.0,
                        "weight": 1,
                    },
                    "sidechain": {
                        "clamp_distance": 10.0,
                        "length_scale": 1.0,
                        "weight": 1,
                    },
                    "eps": eps,
                    "weight": 1.0,
                },
                "supervised_chi": {
                    "chi_weight": 1,
                    "angle_norm_weight": 0.01,
                    "eps": eps,  # 1e-6,
                    "weight": 1.0,
                },
                "violation": {
                    "violation_tolerance_factor": 12.0, #12.0
                    "clash_overlap_tolerance": 1,
                    "eps": eps,  # 1e-6,
                    "weight": 1,
                },
            }
        })


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

def fbb_backbone_loss(
    backbone_rigid_tensor: torch.Tensor,
    backbone_rigid_mask: torch.Tensor,
    traj: torch.Tensor,
    use_clamped_fape: Optional[torch.Tensor] = None,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:



    pred_aff = Rigid.from_tensor_7(traj) #.view(*batch_dims, -1, 7)
    pred_aff = Rigid(Rotation(rot_mats=pred_aff.get_rots().get_rot_mats(), quats=None),pred_aff.get_trans(),)




    # pred_aff = pred_aff.view(*batch_dims, -1, 4, 4)
    # sidechain_frames = Rigid.from_tensor_4x4(sidechain_frames)

    # DISCREPANCY: DeepMind somehow gets a hold of a tensor_7 version of
    # backbone tensor, normalizes it, and then turns it back to a rotation
    # matrix. To avoid a potentially numerically unstable rotation matrix
    # to quaternion conversion, we just use the original rotation matrix
    # outright. This one hasn't been composed a bunch of times, though, so
    # it might be fine.
    gt_aff = backbone_rigid_tensor

    fape_loss = compute_fape(
        pred_aff,
        gt_aff,
        backbone_rigid_mask,
        pred_aff.get_trans(),
        gt_aff.get_trans(),
        backbone_rigid_mask,
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
        chi_mask[...,  :, :], sq_chi_error, dim=(-1,-2 )
    )



    # compute in degree
    sq_chi_error_degree=torch.clip(sq_chi_error,0+eps,4-eps)
    angles_diffs=180*torch.arccos((2-sq_chi_error_degree)/2)/torch.pi
    #adpa=angles_diffs.detach().cpu().numpy()[0]
    angles_diffs_mean=masked_mean(chi_mask, angles_diffs, dim=(-1, -2))

    loss = chi_weight * (sq_chi_loss)


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

    return loss,masked_mean(chi_mask,angles_diffs,dim=(0,1)),(chi_mask,angles_diffs)

def supervised_chi_huberloss(
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

    sq_chi_error_huber=HuberLoss(pred_angles,true_chi)
    sq_chi_error_shift_huber=HuberLoss(pred_angles,true_chi_shifted)
    sq_chi_error_huber = torch.minimum(sq_chi_error_huber, sq_chi_error_shift_huber)

    sq_chi_huberloss = masked_mean(
        chi_mask[...,  :, :], sq_chi_error_huber, dim=(-1,-2 )
    )


    loss = chi_weight * sq_chi_huberloss


    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss


def supervised_chi_acc(
        angles_sin_cos: torch.Tensor,
        unnormalized_angles_sin_cos: torch.Tensor,
        aatype: torch.Tensor,
        seq_mask: torch.Tensor,
        chi_mask: torch.Tensor,
        gt_angles_sin_cos: torch.Tensor,
        chi_weight: float,
        angle_norm_weight: float,
        eps=1e-12,
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
        chi_mask[...,  :, :], sq_chi_error, dim=(-1,-2 )
    )



    # compute in degree
    sq_chi_error_degree=torch.clip(sq_chi_error,0+eps,4-eps)
    angles_diffs=180*torch.arccos((2-sq_chi_error_degree)/2)/torch.pi
    #adpa=angles_diffs.detach().cpu().numpy()[0]
    angles_diffs_mean=masked_mean(chi_mask, angles_diffs, dim=(-1, -2))

    loss = chi_weight * (sq_chi_loss)


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

    return   chi_mask.squeeze(0),angles_diffs.squeeze(0)# masked_mean(chi_mask,angles_diffs,dim=(0,1))  #


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
        bb_angle_mask[...,  :], sq_chi_error, dim=(-1,-2 )
    )
    # C=sq_chi_loss.detach().cpu().numpy()



    # angle in degrees
    sq_chi_error_b = torch.clamp(sq_chi_error.detach(), min=0, max=2)
    angles_diffs=180*torch.arccos((2-sq_chi_error_b)/2)/torch.pi

    angles_diff=   masked_mean(
        bb_angle_mask[...,  :], angles_diffs, dim=(0,1)
    )

    loss = bb_torsion_weight * sq_chi_loss#+bb_torsion_weight*angles_diff

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss,angles_diff



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
    gt_frames=gt_frames.to_tensor_7()
    pred_quanteron=pred_frames[:,:,:4]
    gt_quanteron=gt_frames[:,:,:4]

    error_q_dist = torch.sqrt(
        torch.sum((pred_quanteron - gt_quanteron) ** 2, dim=-1) + eps
    )
    normed_q_error = error_q_dist* gt_frames_mask
    normed_q_error=torch.sum(normed_q_error,-1)

    normed_q_error = (
        normed_q_error / (eps + torch.sum(gt_frames_mask, dim=-1))[..., None]
    )
    #norm in batch dims
    normed_q_error =torch.mean(normed_q_error)

    pred_trans=pred_frames[:,:,4:]
    gt_trans=gt_frames[:,:,4:]
    error_t_dist = torch.sqrt(
        torch.sum((pred_trans - gt_trans) ** 2, dim=-1) + eps
    )
    normed_t_error =error_t_dist* gt_frames_mask
    normed_t_error=torch.sum(normed_t_error,-1)
    normed_t_error = (
        normed_t_error / (eps + torch.sum(gt_frames_mask, dim=-1))[..., None]
    )
    normed_t_error = torch.mean( normed_t_error )


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

        seq_mask=gtframes_mask


        chi_loss,angle_diff_chi,chi_error=supervised_chi_loss(
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
            return chi_loss, bb_angle_loss,angle_diff_bb,angle_diff_chi,chi_error
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


def loss(out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
         fae_t_f: float,
         fae_q_f: float,
         aa_f: float,
            angle_f:float,**kwargs
) -> torch.Tensor:
    bb_fae_loss_q,bb_fae_loss_t = Frame_Aligned_error( pred_frames=out["frames"],gt_frames=batch["gtframes"][...,:,0,:,:],gt_frames_mask=batch['gtframes_mask'][...,:,0])


    bb_angle_loss,angle_diff_bb,chi_errors = torsion_angle_loss(**out, **batch,requires_degree=True,just_bb=True,**kwargs )





    # mask X
    Xmask=batch['gt_aatype']!=20
    seq_mask=Xmask*batch['gtframes_mask'][...,:,0]
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

    gt_ca=Rigid.from_tensor_4x4(batch['gtframes'][...,:,0,:,:]).get_trans().detach()
    pred_ca=out['frames'][:,:,4:].detach()
    ca_lddt=lddt(pred_ca,gt_ca,true_points_mask=batch['gtframes_mask'][...,:,0].unsqueeze(-1))
    ca_lddt=torch.mean(ca_lddt)*100

    Angle_accs = angle_diff_bb.detach()

    result.update({
        'finial_loss': loss.detach().cpu(),
        'ca_lddt': ca_lddt.cpu(),
        'Angle_accs': Angle_accs.cpu(),

    })


    return loss,result



def sidechain_loss(
    sidechain_frames: torch.Tensor,
    sidechain_atom_pos: torch.Tensor,
    rigidgroups_gt_frames: torch.Tensor,
    rigidgroups_alt_gt_frames: torch.Tensor,
    rigidgroups_gt_exists: torch.Tensor,
    renamed_atom14_gt_positions: torch.Tensor,
    renamed_atom14_gt_exists: torch.Tensor,
    alt_naming_is_better: torch.Tensor,
    clamp_distance: float = 10.0,
    length_scale: float = 10.0,
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:
    renamed_gt_frames = (
        1.0 - alt_naming_is_better[..., None, None, None]
    ) * rigidgroups_gt_frames + alt_naming_is_better[
        ..., None, None, None
    ] * rigidgroups_alt_gt_frames

    # Steamroll the inputs
    # sidechain_frames = sidechain_frames[-1]
    batch_dims = sidechain_frames.shape[:-4]
    lengths = sidechain_frames.shape[:-3]
    sidechain_frames = sidechain_frames.view(*batch_dims, -1, 4, 4)
    sidechain_frames = Rigid.from_tensor_4x4(sidechain_frames)
    renamed_gt_frames = renamed_gt_frames.view(*batch_dims, -1, 4, 4)
    renamed_gt_frames = Rigid.from_tensor_4x4(renamed_gt_frames)
    rigidgroups_gt_exists = rigidgroups_gt_exists.reshape(*batch_dims, -1)
    # sidechain_atom_pos = sidechain_atom_pos[-1]
    sidechain_atom_pos = sidechain_atom_pos.view(*batch_dims, -1, 3)
    renamed_atom14_gt_positions = renamed_atom14_gt_positions.view(
         *batch_dims, -1, 3
     )
    renamed_atom14_gt_exists = renamed_atom14_gt_exists.view(*batch_dims, -1)

    fape = compute_fape(
        sidechain_frames,
        renamed_gt_frames,
        rigidgroups_gt_exists,
        sidechain_atom_pos,
        renamed_atom14_gt_positions,
        renamed_atom14_gt_exists,
        l1_clamp_distance=clamp_distance,
        length_scale=length_scale,
        eps=eps,
    )
    # Average over the batch dimension
    # seqmask=rigidgroups_gt_exists[:,:,0]
    # fape_loss=masked_mean(seqmask,fape,dim=(0,1))
    fape_loss=torch.mean(fape)

    return fape_loss


def fape_loss(
        out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        config: mlc.ConfigDict,
) -> torch.Tensor:



    sc_loss = sidechain_loss(
        out["sidechain_frames"],
        out["positions"],
        **{**batch, **config.sidechain},
    )

    loss = config.sidechain.weight * sc_loss




    return loss


def compute_renamed_ground_truth(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:
    """
    Find optimal renaming of ground truth based on the predicted positions.
    Alg. 26 "renameSymmetricGroundTruthAtoms"
    This renamed ground truth is then used for all losses,
    such that each loss moves the atoms in the same direction.
    Args:
      batch: Dictionary containing:
        * atom14_gt_positions: Ground truth positions.
        * atom14_alt_gt_positions: Ground truth positions with renaming swaps.
        * atom14_atom_is_ambiguous: 1.0 for atoms that are affected by
            renaming swaps.
        * atom14_gt_exists: Mask for which atoms exist in ground truth.
        * atom14_alt_gt_exists: Mask for which atoms exist in ground truth
            after renaming.
        * atom14_atom_exists: Mask for whether each atom is part of the given
            amino acid type.
      atom14_pred_positions: Array of atom positions in global frame with shape
    Returns:
      Dictionary containing:
        alt_naming_is_better: Array with 1.0 where alternative swap is better.
        renamed_atom14_gt_positions: Array of optimal ground truth positions
          after renaming swaps are performed.
        renamed_atom14_gt_exists: Mask after renaming swap is performed.
    """

    pred_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., None, :, None, :]
                - atom14_pred_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    atom14_gt_positions = batch["atom14_gt_positions"]
    gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_gt_positions[..., None, :, None, :]
                - atom14_gt_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    atom14_alt_gt_positions = batch["atom14_alt_gt_positions"]
    alt_gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_alt_gt_positions[..., None, :, None, :]
                - atom14_alt_gt_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    lddt = torch.sqrt(eps + (pred_dists - gt_dists) ** 2)
    alt_lddt = torch.sqrt(eps + (pred_dists - alt_gt_dists) ** 2)

    atom14_gt_exists = batch["atom14_gt_exists"]
    atom14_atom_is_ambiguous = batch["atom14_atom_is_ambiguous"]
    mask = (
        atom14_gt_exists[..., None, :, None]
        * atom14_atom_is_ambiguous[..., None, :, None]
        * atom14_gt_exists[..., None, :, None, :]
        * (1.0 - atom14_atom_is_ambiguous[..., None, :, None, :])
    )

    per_res_lddt = torch.sum(mask * lddt, dim=(-1, -2, -3))
    alt_per_res_lddt = torch.sum(mask * alt_lddt, dim=(-1, -2, -3))

    fp_type = atom14_pred_positions.dtype
    alt_naming_is_better = (alt_per_res_lddt < per_res_lddt).type(fp_type)

    renamed_atom14_gt_positions = (
        1.0 - alt_naming_is_better[..., None, None]
    ) * atom14_gt_positions + alt_naming_is_better[
        ..., None, None
    ] * atom14_alt_gt_positions

    renamed_atom14_gt_mask = (
        1.0 - alt_naming_is_better[..., None]
    ) * atom14_gt_exists + alt_naming_is_better[..., None] * batch[
        "atom14_alt_gt_exists"
    ]

    return {
        "alt_naming_is_better": alt_naming_is_better,
        "renamed_atom14_gt_positions": renamed_atom14_gt_positions,
        "renamed_atom14_gt_exists": renamed_atom14_gt_mask,
    }


def Fape_loss( out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],

) -> torch.Tensor:



    batch.update(
        compute_renamed_ground_truth(
            batch,
            out["positions"],
        ))

    fape=fape_loss(
        out,
        batch,
        loss_config.loss.fape
     )
    return fape


def RMSD_CA_error(backbone_rigid_tensor: torch.Tensor,
               backbone_rigid_mask: torch.Tensor,
               traj: torch.Tensor,

               loss_unit_distance: float = 10.0,
               eps: float = 1e-4,
               **kwargs,
               ) -> torch.Tensor:
    pred_aff = Rigid.from_tensor_7(traj)
    pred_aff = Rigid(
        Rotation(rot_mats=pred_aff.get_rots().get_rot_mats(), quats=None),
        pred_aff.get_trans(),
    )
    gt_aff = backbone_rigid_tensor
    pred_ca = pred_aff.get_trans()
    gt_ca = gt_aff.get_trans()

    error = torch.sqrt(
        torch.sum((pred_ca - gt_ca) ** 2, dim=-1) + eps
    )

    errors = masked_mean(backbone_rigid_mask, error, dim=(0, 1))

    return errors


def FBB_auxloss(out: Dict[str, torch.Tensor],
             batch: Dict[str, torch.Tensor],
             fape_f: float,
             angle_f: float,
                rmsd_f: float,**kwargs
             ) -> torch.Tensor:
    batch['backbone_rigid_tensor'] = batch['gtframes']
    batch['backbone_rigid_mask'] = batch['gtframes_mask']

    # out["frames"]=batch['gtframes'].to_tensor_7()

    bbfape = fbb_backbone_loss(traj=out["frames"], **{**batch, **loss_config.loss.fape.backbone}, )

    rmsd=RMSD_error(traj=out["frames"], **{**batch,}, )

    bb_angle_loss, angle_diff_bb = torsion_angle_loss(**out, **batch, requires_degree=True, just_bb=True, **kwargs)

    #aa loss
    # mask X
    Xmask=batch['gt_aatype']!=20
    seq_mask=Xmask*batch['gtframes_mask']
    gt_aatype=batch['gt_aatype']*Xmask
    score, bb_loss_av=aa_loss_smoothed(
        gt_aatype=gt_aatype,
        logits=out['logits'],
        seq_mask=seq_mask,**kwargs,)


    loss =fape_f* bbfape + angle_f * bb_angle_loss+rmsd_f*rmsd#+bb_loss_av




    return loss






def FBB_loss(out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
         fape_f: float,

         aa_f: float,
            angle_f:float,  rmsd_f: float,**kwargs
) -> torch.Tensor:


    batch['backbone_rigid_tensor']=batch['gtframes']
    batch['backbone_rigid_mask']=batch['gtframes_mask']

    bbfape = fbb_backbone_loss(traj=out["frames"], **{**batch, **loss_config.loss.fape.backbone}, )
    rmsd=RMSD_error(traj=out["frames"], **{**batch,}, )
    bb_angle_loss,angle_diff_bb = torsion_angle_loss(**out, **batch,requires_degree=True,just_bb=True,**kwargs )

    # mask X
    Xmask=batch['gt_aatype']!=20
    seq_mask=Xmask*batch['gtframes_mask']
    gt_aatype=batch['gt_aatype']*Xmask


    score, bb_loss_av=aa_loss_smoothed(
        gt_aatype=gt_aatype,
        logits=out['logits'],
        seq_mask=seq_mask,**kwargs,)



    loss = fape_f*bbfape+angle_f* bb_angle_loss+rmsd_f*rmsd + aa_f * bb_loss_av

    result={
        'ca_rmsd_loss': rmsd.detach(),
        'bb_fape_loss': bbfape.detach(),
        'bb_angle_loss':bb_angle_loss.detach(),
        'aa_loss':bb_loss_av.detach(),
    }

    gt_ca=batch['gtframes'].get_trans().detach()
    pred_ca=out['positions'][:,:,1,:].detach()
    ca_lddt=lddt(pred_ca,gt_ca,true_points_mask=batch['gtframes_mask'].unsqueeze(-1))
    ca_lddt=torch.mean(ca_lddt)*100

    Angle_accs = angle_diff_bb.detach()

    result.update({
        'finial_loss': loss.detach(),
        'ca_lddt': ca_lddt,
        'Angle_accs': Angle_accs,
    })


    return loss,result



def RMSD(pred,target,mask,eps=1e-12):
    dis=pred.type(torch.float)-target.type(torch.float)
    norm_error = torch.sqrt(
        torch.sum(dis ** 2, dim=-1) + eps
    )
    # TESTMASK=mask.numpy()
    # test=norm_error.numpy()*mask.numpy()
    # print(pred[13])
    # print(target[13])

    _loss = masked_mean(
        mask[...,  :].type(torch.float), norm_error, dim=(1)
    )
    # _loss=torch.mean(norm_error,dim=-1)
    return _loss





# def sidechain_huber_loss(
#
#     sidechain_atom_pos: torch.Tensor,
#
#     renamed_atom14_gt_positions: torch.Tensor,
#     renamed_atom14_gt_exists: torch.Tensor,
#
#     clamp_distance: float = 10.0,
#     length_scale: float = 10.0,
#     eps: float = 1e-4,
#     **kwargs,
# ) -> torch.Tensor:
#     sidechain_atom_pos=sidechain_atom_pos[..., 5:,:]
#     renamed_atom14_gt_positions=renamed_atom14_gt_positions[..., 5:,:]
#     renamed_atom14_gt_exists=renamed_atom14_gt_exists[..., 5:]
#
#     sidechain_atom_pos=sidechain_atom_pos*renamed_atom14_gt_exists[...,None]
#     renamed_atom14_gt_positions=renamed_atom14_gt_positions*renamed_atom14_gt_exists[...,None]
#
#     # huberloss_va=HuberLoss(sidechain_atom_pos,renamed_atom14_gt_positions)
#     # huberloss_va=torch.sum(huberloss_va,dim=-1)
#     # huberloss_sidechain=masked_mean(renamed_atom14_gt_exists,huberloss_va,dim=(0,1,2))
#
#
#     RMSD_sidechain=RMSD(sidechain_atom_pos,renamed_atom14_gt_positions,renamed_atom14_gt_exists)
#
#
#
#     return RMSD_sidechain

# def huber_loss(
#         out: Dict[str, torch.Tensor],
#         batch: Dict[str, torch.Tensor],
#
# ) -> torch.Tensor:
#
#
#
#     RMSD_sidechain = sidechain_huber_loss(
#
#         out["positions"],
#         **batch,
#     )
#
#     return RMSD_sidechain

def atoms_huber_loss(out: Dict[str, torch.Tensor],
              batch: Dict[str, torch.Tensor],

              ) -> torch.Tensor:
    batch.update(
        compute_renamed_ground_truth(
            batch,
            out["positions"],
        ))

    RMSD_sidechain  = huber_loss(
        out,
        batch,
    )
    return RMSD_sidechain



def Repacker_Aux_loss(out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],

             fape_f: float,

    angle_f:float,
   **kwargs) -> torch.Tensor:
    newbatch={
        'gtframes':Rigid.from_tensor_4x4(batch['rigidgroups_gt_frames'][:,:,0,:,:]),
        'gtframes_mask': batch['rigidgroups_gt_exists'][:, :, 0],
        'gt_aatype': batch['aatype'],
        'gt_angles_sin_cos': batch['gt_angles_sin_cos'],
        'gt_angles_mask': batch['gt_angles_mask'],
    }

    #bb_fae_loss_q,bb_fae_loss_t = Frame_Aligned_error( pred_frames=out["frames"],gt_frames=newbatch["gtframes"],gt_frames_mask=newbatch['gtframes_mask'])

    batch['backbone_rigid_tensor']=batch['rigidgroups_gt_frames'][:,:,0,:,:]
    batch['backbone_rigid_mask']=batch['rigidgroups_gt_exists'][:, :, 0]
    #bbfape=backbone_loss(traj=out["frames"],**{**batch, **loss_config.loss.fape.backbone},)

    bfloss=bf_loss(out['pred_b_factors'],batch['b_factors'])
    # framesloss=torch.sqrt(
    #     torch.sum((out["sidechain_frames"]- batch["rigidgroups_gt_frames"]) ** 2, dim=(-1,-2)))
    # framesloss=masked_mean( batch["rigidgroups_gt_exists"],framesloss,dim=(0,1))
    #
    # framesloss=torch.mean(framesloss)*0.2


    fapeloss=Fape_loss(out,batch)

    chi_loss,bb_angle_loss = torsion_angle_loss(**out, **newbatch,**kwargs )
    # E=energy_side(batch['aatype'],out["angles_sin_cos"],dunbrak,batch['gt_angles_mask'])/20  #out["angles_sin_cos"]

    loss = angle_f * (chi_loss+bb_angle_loss)+fape_f*fapeloss+bfloss*0.1#+E



    return loss


def violationloss(out,batch):

    out["violation"] = find_structural_violations(
        batch,
        out["positions"],
        **loss_config.loss.violation,
    )

    violation= violation_loss(
        out["violation"],
        **batch,
    )
    return violation

def bf_loss(pred_b_factors:torch.tensor,
            b_factors:torch.tensor,
            ):

    gt_bf=b_factors[:,:,[3,5,6,7,8,9,10,11,12,13]].detach()
    bfmask=(gt_bf!=0).detach()
    bfloss=torch.sqrt((pred_b_factors - gt_bf) ** 2+eps)
    bfloss = torch.clamp(bfloss, min=0, max=150)


    bflosss=masked_mean(bfmask,bfloss,dim=(-1,-2))
    bflosss=torch.mean(bflosss )

    return bflosss


def energy_side(aatype,anlges,dunrack,chis_mask):

    angles_tan=anlges[...,0] / (anlges[...,1]+eps)
    angles_tan=torch.clip(angles_tan,-1,1)
    anlges=180 * torch.arctan(angles_tan/ torch.pi)

    dunrack=dunrack.to(aatype.device)
    energy_mean=fake_energy(aatype,anlges,dunrack,chis_mask[...,3:])

    return energy_mean


def Repackerloss(out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
         fape_f: float,
            angle_f:float,**kwargs
) -> torch.Tensor:
    newbatch={
        'gtframes':Rigid.from_tensor_4x4(batch['rigidgroups_gt_frames'][:,:,0,:,:]),
        'gtframes_mask': batch['rigidgroups_gt_exists'][:, :, 0],
        'gt_aatype': batch['aatype'],
        'gt_angles_sin_cos': batch['gt_angles_sin_cos'],
        'gt_angles_mask': batch['gt_angles_mask'],
    }

    batch['backbone_rigid_tensor']=batch['rigidgroups_gt_frames'][:,:,0,:,:]
    batch['backbone_rigid_mask']=batch['rigidgroups_gt_exists'][:, :, 0]
    bbfape=backbone_loss(traj=out["frames"],**{**batch, **loss_config.loss.fape.backbone},)
    chi_loss,bb_angle_loss,angle_diff_bb,angle_diff_chi,chi_errors = torsion_angle_loss(**out, **newbatch,requires_degree=True,**kwargs )


    # framesloss=torch.sqrt(torch.sum((out["sidechain_frames"] - batch["rigidgroups_gt_frames"]) ** 2, dim=(-1,-2)))
    # framesloss=masked_mean( batch["rigidgroups_gt_exists"],framesloss,dim=(0,1 ))

    fapeloss=Fape_loss(out,batch)
    # framesloss=torch.mean(framesloss)

    bfloss=bf_loss(out['pred_b_factors'],batch['b_factors'])

    # violations=violationloss(out,batch)

    #fake side energy
    #E=energy_side(batch['aatype'],out["angles_sin_cos"],dunbrak,batch['gt_angles_mask'])/20

    # RMSD_sidechain =atoms_huber_loss(out,batch)25222


    loss = bbfape+ angle_f * (chi_loss+bb_angle_loss)+fape_f*fapeloss+bfloss*0.1 #+ RMSD_sidechain#+E

    result={
        'bfloss':bfloss.detach().clone(),
        'chi_loss': chi_loss.detach().clone(),
        'bb_angle_loss': bb_angle_loss.detach().clone(),
        # 'violations':violations.detach().clone(),
        'fapeloss':fapeloss.detach().clone(),

    }

    gt_ca=newbatch['gtframes'].get_trans().detach().clone()
    pred_ca=out['frames'][:,:,4:].detach().clone()
    ca_lddt=lddt(pred_ca,gt_ca,true_points_mask=newbatch['gtframes_mask'].unsqueeze(-1))
    ca_lddt=torch.mean(ca_lddt)*100

    Angle_accs=torch.cat((angle_diff_bb,angle_diff_chi),dim=0).detach().clone().type(torch.int)



    result.update({
        'finial_loss': loss.detach().clone(),
        'ca_lddt': ca_lddt,
        'Angle_accs': Angle_accs,
        'chi_error': chi_errors[1].cpu(),
        'chi_mask': chi_errors[0].cpu(),
    })


    return loss,result

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