import torch
import torch.nn as nn
import torch.nn.functional as F
from model.losses import *

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"



class VAELoss(nn.Module):
    def __init__(self, kl_p=0.0002):
        super(VAELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.kl_loss = KLLoss()
        self.kl_p = kl_p

    def forward(self, gt_emotion, pred_emotion, distribution):
        rec_loss = self.mse(pred_emotion, gt_emotion)

        mu_ref = torch.zeros_like(distribution[0].loc).to(gt_emotion.get_device())
        scale_ref = torch.ones_like(distribution[0].scale).to(gt_emotion.get_device())
        distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)

        kld_loss = 0
        for t in range(len(distribution)):
            kld_loss += self.kl_loss(distribution[t], distribution_ref)
        kld_loss = kld_loss / len(distribution)

        loss = rec_loss + self.kl_p * kld_loss
        # loss = rec_loss + kld_loss
        

        return loss, rec_loss, kld_loss

    def __repr__(self):
        return "VAELoss()"



def div_loss(Y_1, Y_2):
    loss = 0.0
    b,t,c = Y_1.shape
    Y_g = torch.cat([Y_1.view(b,1,-1), Y_2.view(b,1,-1)], dim = 1)
    for Y in Y_g:
        dist = F.pdist(Y, 2) ** 2
        loss += (-dist /  100).exp().mean()
    loss /= b
    return loss

class SmoothLoss(nn.Module):
    def __init__(self, k =0.1):
        super(SmoothLoss, self).__init__()
        self.sml1 = nn.SmoothL1Loss(reduce=True, size_average=True)
        self.k = k

    def forward(self, x):
        loss = self.sml1((x[:, 2:, 52:] - x[:, 1:-1, 52:]),
                     (x[:, 1:-1, 52:] - x[:, :-2, 52:])) + \
                self.k * self.sml1(
                    (x[:, 2:, :52] - x[:, 1:-1, :52]),
                    (x[:, 1:-1, :52] - x[:, :-2, :52]))
        return loss

    def __repr__(self):
        return "SmoothLoss"


# # ================================ BeLFUSION losses ====================================

# def MSELoss_AE_v2(prediction, target, target_coefficients, mu, logvar, coefficients_3dmm, 
#                   w_mse=1, w_kld=1, w_coeff=1, 
#                   **kwargs):
#     # loss for autoencoder. prediction and target have shape of [batch_size, seq_length, features]
#     assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
#     assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, seq_length, features]"
#     batch_size = prediction.shape[0]

#     # join last two dimensions of prediction and target
#     prediction = prediction.reshape(prediction.shape[0], -1)
#     target = target.reshape(target.shape[0], -1)
#     coefficients_3dmm = coefficients_3dmm.reshape(coefficients_3dmm.shape[0], -1)
#     target_coefficients = target_coefficients.reshape(target_coefficients.shape[0], -1)

#     MSE = ((prediction - target) ** 2).mean()
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
#     COEFF = ((coefficients_3dmm - target_coefficients) ** 2).mean()

#     loss_r = w_mse * MSE + w_kld * KLD + w_coeff * COEFF
#     return {"loss": loss_r, "mse": MSE, "kld": KLD, "coeff": COEFF}


# def MSELoss(prediction, target, reduction="mean", **kwargs):
#     # prediction has shape of [batch_size, num_preds, features]
#     # target has shape of [batch_size, num_preds, features]
#     assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
#     assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

#     # manual implementation of MSE loss
#     loss = ((prediction - target) ** 2).mean(axis=-1)
    
#     # reduce across multiple predictions
#     if reduction == "mean":
#         loss = torch.mean(loss)
#     elif reduction == "min":
#         loss = loss.min(axis=-1)[0].mean()
#     else:
#         raise NotImplementedError("reduction {} not implemented".format(reduction))
#     return loss


# def L1Loss(prediction, target, reduction="mean", **kwargs):
#     # prediction has shape of [batch_size, num_preds, features]
#     # target has shape of [batch_size, num_preds, features]
#     assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
#     assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

#     # manual implementation of L1 loss
#     loss = (torch.abs(prediction - target)).mean(axis=-1)
    
#     # reduce across multiple predictions
#     if reduction == "mean":
#         loss = torch.mean(loss)
#     elif reduction == "min":
#         loss = loss.min(axis=-1)[0].mean()
#     else:
#         raise NotImplementedError("reduction {} not implemented".format(reduction))
#     return loss
