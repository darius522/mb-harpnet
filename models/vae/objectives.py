# objectives of choice

import torch
import torch.nn.functional as F
import torch.distributions as dist
from numpy import prod
import math

def has_analytic_kl(type_p, type_q):
    return (type_p, type_q) in torch.distributions.kl._KL_REGISTRY

def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))

def kl_divergence(p, q, samples=None):
    if has_analytic_kl(type(p), type(q)):
        return dist.kl_divergence(p, q)
    else:
        if samples is None:
            K = 10
            samples = p.rsample(torch.Size([K])) if p.has_rsample else p.sample(torch.Size([K]))
        # ent = p.entropy().unsqueeze(0) if hasattr(p, 'entropy') else -p.log_prob(samples)
        ent = -p.log_prob(samples)
        return (-ent - q.log_prob(samples)).mean(0)

# helper to vectorise computation
def compute_microbatch_split(x, K):
    """ Checks if batch needs to be broken down further to fit in memory. """
    B = x.size(0)
    S = int(1e8 / (K * prod(x.size()[1:])))  # float heuristic for 12Gb cuda memory
    assert (S > 0), "Cannot fit individual data in memory, consider smaller K"
    return min(B, S)

# only should have one training objective
def decomp_objective(model, x, K=1, beta=1.0, alpha=0.0, regs=None, components=False):
    """Computes E_{p(x)}[ELBO_{\alpha,\beta}] """
    qz_x, px_z, zs = model(x, K)
    # torch.Size([64, 1, 32, 32]) torch.Size([64, 10]) torch.Size([1, 64, 1, 32, 32]) torch.Size([1, 64, 10])
    # torch.Size([20, 2, 1024]) torch.Size([20, 10, 1]) torch.Size([20, 2, 1024]) torch.Size([1, 20, 10, 1])
    #print(x.shape, qz_x.batch_shape, px_z.batch_shape, zs.shape)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:1], -1).sum(-1).unsqueeze(0)
    pz = model.pz(*model.pz_params)
    kld = kl_divergence(qz_x, pz, samples=zs.squeeze(-1)).sum(-1)
    reg = (regs(pz.sample(torch.Size([x.size(0)])).view(-1, zs.size(-1)), zs.squeeze(0)) if regs.samples else regs(pz, qz_x)) \
        if regs else torch.tensor(0)
    obj = lpx_z - (beta * kld) - (alpha * reg)
    return obj.sum() if not components else (obj.sum(), lpx_z.sum(), kld.sum(), reg.sum())

# this is only used for test eval
def _iwae_objective_vec(model, x, K=1, **kwargs):
    """Helper for IWAE estimate for log p_\theta(x) -- full vectorisation."""
    qz_x, px_z, zs = model(x, K)
    lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1).sum(-1)
    lqz_x = qz_x.log_prob(zs).sum(-1)
    return lpz + lpx_z - lqz_x

def iwae_objective(model, x, K=1, **kwargs):
    """Computes an importance-weighted ELBO estimate for log p_\theta(x)
    Iterates over the batch as necessary.
    """
    S = compute_microbatch_split(x, K)
    lw = torch.cat([_iwae_objective_vec(model, _x.contiguous(), K) for _x in x.split(S)], 1)  # concat on batch
    return log_mean_exp(lw).sum()