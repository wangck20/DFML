import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from pytorch_metric_learning import miners, losses

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss

# We use PyTorch Metric Learning library for the following codes.
# Please refer to "https://github.com/KevinMusgrave/pytorch-metric-learning" for details.
#class Proxy_NCA(torch.nn.Module):
#    def __init__(self, nb_classes, sz_embed, scale=32):
#        super(Proxy_NCA, self).__init__()
#        self.nb_classes = nb_classes
#        self.sz_embed = sz_embed
#        self.scale = scale
#        self.loss_func = losses.ProxyNCALoss(num_classes = self.nb_classes, embedding_size = self.sz_embed, softmax_scale = self.scale).cuda()

#    def forward(self, embeddings, labels):
#        loss = self.loss_func(embeddings, labels)
#        return loss

class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed):
        super(Proxy_NCA, self).__init__()
        self.num_proxies = nb_classes
        self.embedding_dim = sz_embed
        self.PROXIES = torch.nn.Parameter(torch.randn(self.num_proxies, self.embedding_dim).cuda() / 8)
        self.all_classes = torch.arange(self.num_proxies)

    def forward(self, embeddings, labels):
        batch       = 3*torch.nn.functional.normalize(embeddings, dim=1)
        PROXIES     = 3*torch.nn.functional.normalize(self.PROXIES, dim=1)
        pos_proxies = torch.stack([PROXIES[pos_label:pos_label+1,:] for pos_label in labels])
        neg_proxies = torch.stack([torch.cat([self.all_classes[:class_label],self.all_classes[class_label+1:]]) for class_label in labels])
        neg_proxies = torch.stack([PROXIES[neg_labels,:] for neg_labels in neg_proxies])
        dist_to_neg_proxies = torch.sum((batch[:,None,:]-neg_proxies).pow(2),dim=-1)
        dist_to_pos_proxies = torch.sum((batch[:,None,:]-pos_proxies).pow(2),dim=-1)
        loss = torch.mean(dist_to_pos_proxies[:,0] + torch.logsumexp(-dist_to_neg_proxies, dim=1))
        return loss

class MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50
        
        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin) 
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class TripletLoss(torch.nn.Module):
    def __init__(self, margin=0.2):
        """
        Basic Triplet Loss as proposed in 'FaceNet: A Unified Embedding for Face Recognition and Clustering'
        Args:
            margin:             float, Triplet Margin - Ensures that positives aren't placed arbitrarily close to the anchor.
                                Similarl, negatives should not be placed arbitrarily far away.
            sampling_method:    Method to use for sampling training triplets. Used for the TupleSampler-class.
        """
        super(TripletLoss, self).__init__()
        self.margin             = margin

    def pdist(self, A, eps = 1e-4):
        """
        Efficient function to compute the distance matrix for a matrix A.

        Args:
            A:   Matrix/Tensor for which the distance matrix is to be computed.
            eps: float, minimal distance/clampling value to ensure no zero values.
        Returns:
            distance_matrix, clamped to ensure no zero values are passed.
        """
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min = 0)
        return res.clamp(min = eps).sqrt()

    def semihardsampling(self, batch, labels):
        """
        This methods finds all available triplets in a batch given by the classes provided in labels, and select
        triplets based on semihard sampling introduced in 'Deep Metric Learning via Lifted Structured Feature Embedding'.

        Args:
            batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.
            labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.
        Returns:
            list of sampled data tuples containing reference indices to the position IN THE BATCH.
        """
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        bs = batch.size(0)
        #Return distance matrix for all elements in batch (BSxBS)
        distances = self.pdist(batch.detach()).detach().cpu().numpy()

        positives, negatives = [], []
        anchors = []
        for i in range(bs):
            l, d = labels[i], distances[i]
            anchors.append(i)
            #1 for batchelements with label l
            neg = labels!=l; pos = labels==l
            #0 for current anchor
            pos[i] = False

            #Find negatives that violate triplet constraint semi-negatives
            neg_mask = np.logical_and(neg,d<d[np.where(pos)[0]].max())
            #Find positives that violate triplet constraint semi-hardly
            pos_mask = np.logical_and(pos,d>d[np.where(neg)[0]].min())

            if pos_mask.sum()>0:
                positives.append(np.random.choice(np.where(pos_mask)[0]))
            else:
                positives.append(np.random.choice(np.where(pos)[0]))

            if neg_mask.sum()>0:
                negatives.append(np.random.choice(np.where(neg_mask)[0]))
            else:
                negatives.append(np.random.choice(np.where(neg)[0]))

        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]
        return sampled_triplets

    def triplet_distance(self, anchor, positive, negative):
        """
        Compute triplet loss.

        Args:
            anchor, positive, negative: torch.Tensor(), resp. embeddings for anchor, positive and negative samples.
        Returns:
            triplet loss (torch.Tensor())
        """
        return torch.nn.functional.relu((anchor-positive).pow(2).sum()-(anchor-negative).pow(2).sum()+self.margin)

    def forward(self, batch, labels):
        """
        Args:
            batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings
            labels:  np.ndarray [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        Returns:
            triplet loss (torch.Tensor(), batch-averaged)
        """
        #Sample triplets to use for training.
        sampled_triplets = self.semihardsampling(batch, labels)
        #Compute triplet loss
        loss             = torch.stack([self.triplet_distance(batch[triplet[0],:],batch[triplet[1],:],batch[triplet[2],:]) for triplet in sampled_triplets])

        return torch.mean(loss)

# class TripletLoss(nn.Module):
#     def __init__(self, margin=0.2, **kwargs):
#         super(TripletLoss, self).__init__()
#         self.margin = margin
#         self.miner = miners.TripletMarginMiner(margin, type_of_triplets = 'semihard')
#         self.loss_func = losses.TripletMarginLoss(margin = self.margin)
        
#     def forward(self, embeddings, labels):
#         hard_pairs = self.miner(embeddings, labels)
#         loss = self.loss_func(embeddings, labels, hard_pairs)
#         return loss

class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss()
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss  

class NTXentLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NTXentLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NTXentLoss()
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss

# class MarginLoss(nn.Module):
#     def __init__(self, nb_classes, margin=0.2, **kwargs):
#         super(MarginLoss, self).__init__()
#         self.margin = margin
#         self.miner = miners.DistanceWeightedMiner()
#         self.loss_func = losses.MarginLoss(margin = self.margin, learn_beta = True, num_classes = nb_classes)

#     def forward(self, embeddings, labels):
#         hard_pairs = self.miner(embeddings, labels)
#         loss = self.loss_func(embeddings, labels, hard_pairs)
#         return loss

class MarginLoss(torch.nn.Module):
    def __init__(self, margin=0.2, nu=0, beta=1.2, nb_classes=100):
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.n_classes = nb_classes
        self.beta = torch.nn.Parameter(torch.ones(self.n_classes).cuda()*beta)
        #self.beta = beta
        self.nu = nu

    def distanceweightedsampling(self, batch, labels, lower_cutoff=0.5, upper_cutoff=1.4):
        """
        This methods finds all available triplets in a batch given by the classes provided in labels, and select
        triplets based on distance sampling introduced in 'Sampling Matters in Deep Embedding Learning'.

        Args:
            batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.
            labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.
            lower_cutoff: float, lower cutoff value for negatives that are too close to anchor embeddings. Set to literature value. They will be assigned a zero-sample probability.
            upper_cutoff: float, upper cutoff value for positives that are too far away from the anchor embeddings. Set to literature value. They will be assigned a zero-sample probability.
        Returns:
            list of sampled data tuples containing reference indices to the position IN THE BATCH.
        """
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        bs = batch.shape[0]

        distances    = self.pdist(batch.detach()).clamp(min=lower_cutoff)



        positives, negatives = [],[]
        labels_visited = []
        anchors = []

        for i in range(bs):
            neg = labels!=labels[i]; pos = labels==labels[i]
            q_d_inv = self.inverse_sphere_distances(batch, distances[i], labels, labels[i])
            #Sample positives randomly
            pos[i] = 0
            positives.append(np.random.choice(np.where(pos)[0]))
            #Sample negatives by distance
            negatives.append(np.random.choice(bs,p=q_d_inv))

        sampled_triplets = [[a,p,n] for a,p,n in zip(list(range(bs)), positives, negatives)]
        return sampled_triplets

    def semihardsampling(self, batch, labels):
        """
        This methods finds all available triplets in a batch given by the classes provided in labels, and select
        triplets based on semihard sampling introduced in 'Deep Metric Learning via Lifted Structured Feature Embedding'.

        Args:
            batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.
            labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.
        Returns:
            list of sampled data tuples containing reference indices to the position IN THE BATCH.
        """
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        bs = batch.size(0)
        #Return distance matrix for all elements in batch (BSxBS)
        distances = self.pdist(batch.detach()).detach().cpu().numpy()

        positives, negatives = [], []
        anchors = []
        for i in range(bs):
            l, d = labels[i], distances[i]
            anchors.append(i)
            #1 for batchelements with label l
            neg = labels!=l; pos = labels==l
            #0 for current anchor
            pos[i] = False

            #Find negatives that violate triplet constraint semi-negatives
            neg_mask = np.logical_and(neg,d<d[np.where(pos)[0]].max())
            #Find positives that violate triplet constraint semi-hardly
            pos_mask = np.logical_and(pos,d>d[np.where(neg)[0]].min())

            if pos_mask.sum()>0:
                positives.append(np.random.choice(np.where(pos_mask)[0]))
            else:
                positives.append(np.random.choice(np.where(pos)[0]))

            if neg_mask.sum()>0:
                negatives.append(np.random.choice(np.where(neg_mask)[0]))
            else:
                negatives.append(np.random.choice(np.where(neg)[0]))

        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]
        return sampled_triplets

    def pdist(self, A):
        """
        Efficient function to compute the distance matrix for a matrix A.

        Args:
            A:   Matrix/Tensor for which the distance matrix is to be computed.
            eps: float, minimal distance/clampling value to ensure no zero values.
        Returns:
            distance_matrix, clamped to ensure no zero values are passed.
        """
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min = 0)
        return res.clamp(min = 0).sqrt()


    def inverse_sphere_distances(self, batch, dist, labels, anchor_label):
        """
        Function to utilise the distances of batch samples to compute their
        probability of occurence, and using the inverse to sample actual negatives to the resp. anchor.

        Args:
            batch:        torch.Tensor(), batch for which the sampling probabilities w.r.t to the anchor are computed. Used only to extract the shape.
            dist:         torch.Tensor(), computed distances between anchor to all batch samples.
            labels:       np.ndarray, labels for each sample for which distances were computed in dist.
            anchor_label: float, anchor label
        Returns:
            distance_matrix, clamped to ensure no zero values are passed.
        """
        bs,dim       = len(dist),batch.shape[-1]

        #negated log-distribution of distances of unit sphere in dimension <dim>
        log_q_d_inv = ((2.0 - float(dim)) * torch.log(dist) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (dist.pow(2))))
        #Set sampling probabilities of positives to zero
        log_q_d_inv[np.where(labels==anchor_label)[0]] = 0

        q_d_inv     = torch.exp(log_q_d_inv - torch.max(log_q_d_inv)) # - max(log) for stability
        #Set sampling probabilities of positives to zero
        q_d_inv[np.where(labels==anchor_label)[0]] = 0

        ### NOTE: Cutting of values with high distances made the results slightly worse.
        # q_d_inv[np.where(dist>upper_cutoff)[0]]    = 0

        #Normalize inverted distance for probability distr.
        q_d_inv = q_d_inv/q_d_inv.sum()
        return q_d_inv.detach().cpu().numpy()

    def forward(self, embeddings, labels):
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        sampled_triplets = self.distanceweightedsampling(embeddings, labels)

        d_ap, d_an = [],[]
        for triplet in sampled_triplets:
            train_triplet = {'Anchor': embeddings[triplet[0],:], 'Positive':embeddings[triplet[1],:], 'Negative':embeddings[triplet[2]]}

            pos_dist = ((train_triplet['Anchor']-train_triplet['Positive']).pow(2).sum()+1e-8).pow(1/2)
            neg_dist = ((train_triplet['Anchor']-train_triplet['Negative']).pow(2).sum()+1e-8).pow(1/2)

            d_ap.append(pos_dist)
            d_an.append(neg_dist)
        d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)

        #Group betas together by anchor class in sampled triplets (as each beta belongs to one class).
        beta = torch.stack([self.beta[labels[triplet[0]]] for triplet in sampled_triplets]).type(torch.cuda.FloatTensor)
        #beta = self.beta

        #Compute actual margin postive and margin negative loss
        pos_loss = torch.nn.functional.relu(d_ap-beta+self.margin)
        neg_loss = torch.nn.functional.relu(beta-d_an+self.margin)

        #Compute normalization constant
        pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).type(torch.cuda.FloatTensor)

        #Actual Margin Loss
        loss = torch.sum(pos_loss+neg_loss) if pair_count==0. else torch.sum(pos_loss+neg_loss)/pair_count

        #(Optional) Add regularization penalty on betas.
        if self.nu: loss = loss + beta_regularisation_loss.type(torch.cuda.FloatTensor)

        return loss
