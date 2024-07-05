import torch
import torch.nn as nn
import torch.nn.functional as F

class SeqCLRLoss(nn.Module):
    def __init__(self, temp=0.1, reduction="batchmean", record=True):
        super().__init__()
        self.reduction = reduction
        self.temp = temp
        self.record = record

    @property
    def last_losses(self):
        return self.losses
    
    # seq2seq contrastive loss function
    def _seqclr_loss(self, features0, features1, n_instances_per_view, n_instances_per_image):
        instances = torch.cat((features0, features1), dim=0) #torch.Size([instance_len, 512]) -> torch.Size([instance_len*2, 512])
        normalized_instances = F.normalize(instances, dim=1)
        similarity_matrix = normalized_instances @ normalized_instances.T #matmul
        similarity_matrix_exp = (similarity_matrix / self.temp).exp_()
        cross_entropy_denominator = similarity_matrix_exp.sum(dim=1) - similarity_matrix_exp.diag() #diff seq: negative, same seq: positive
        cross_entropy_nominator = torch.cat((
            similarity_matrix_exp.diagonal(offset=n_instances_per_view)[:n_instances_per_view],
            similarity_matrix_exp.diagonal(offset=-n_instances_per_view)
        ), dim=0)
        cross_entropy_similarity = cross_entropy_nominator / cross_entropy_denominator
        loss = -cross_entropy_similarity.log()

        if self.reduction == "batchmean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean_instances_per_image":
            loss = loss.sum() / n_instances_per_image
        return loss
    
    # seq2seq contrastive loss function
    def cosine_similarity(self, x, y):
        return torch.sum(x * y, dim=-1) / (torch.norm(x, dim=-1) * torch.norm(y, dim=-1))

    def nce_loss(self, ua, ub, U, tau=0.1):
        # ua and ub are the anchor and positive representations, respectively
        # U is the set of all representations (including ua and ub)
        similarities = torch.stack([self.cosine_similarity(ua, u) for u in U])
        exp_similarities = torch.exp(similarities / tau)
        exp_positive_similarity = torch.exp(self.cosine_similarity(ua, ub) / tau)
        return -torch.log(exp_positive_similarity / torch.sum(exp_similarities))

    def contrastive_loss(self, z_a, z_b, U):
        loss = 0.0
        # Loop over all positive pairs
        for i in range(len(z_a)):
            # Compute NCE loss for each positive pair (za_i, zb_i)
            loss += self.nce_loss(z_a[i], z_b[i], U)
            loss += self.nce_loss(z_b[i], z_a[i], U)
        return loss / (2 * len(z_a))  # Normalize by the number of pairs
    
    
    def forward(self, features0, features1, *args, **kwargs):
        seqclr_loss = 0

        n_instances_per_image = features0.shape[1]
        n_instances_per_view = features1.shape[0] * n_instances_per_image

        # 여기 수정
        features0 = torch.flatten(features0, start_dim=0, end_dim=1)
        features1 = torch.flatten(features1, start_dim=0, end_dim=1)
        U = torch.cat([features0, features1], dim=0)
        
        seqclr_loss += self._seqclr_loss(features0, features1, n_instances_per_view, n_instances_per_image)
        # seqclr_loss += self.contrastive_loss(features0, features1, U)

        return seqclr_loss                           