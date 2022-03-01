from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from readData import *
from evaluation import *
from util import *

device_no = '0'
device = torch.device(f'cuda:{device_no}' if torch.cuda.is_available() else 'cpu')
if device == f'cuda:{device_no}':
    torch.cuda.manual_seed(10)


class VectorEmbedding(nn.Module):
    def __init__(self,latent_dim,input_dim):
        super(VectorEmbedding,self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        activate = nn.Tanh()
        self.enc = nn.Sequential(
            nn.Linear(self.input_dim,500),
            activate,
            nn.Linear(500,500),
            activate,
            nn.Linear(500, 2000),
            activate,
            nn.Linear(2000, self.latent_dim),
        )

    def forward(self, x):
        return self.enc(x)


class FARE(nn.Module):
    def __init__(self, latent_dim,input_dim, M=100,alpha=100,useLabel=True):
        super(FARE, self).__init__()
        self.latent_dim = latent_dim
        self.M = M
        self.embedding = VectorEmbedding(latent_dim,input_dim)
        self.contrastive = ConstraintContrastiveLoss(alpha=alpha,M=M,useLabel=useLabel)
        self.useLabel = useLabel

    def forward(self, x, cluster_label, x_supervise, label):
        encoded = self.embedding(x)
        if self.useLabel:
            embedding_supervise = self.embedding(x_supervise)
            return self.contrastive(encoded,cluster_label,embedding_supervise,label)
        else:
            return self.contrastive(encoded, cluster_label, None, label)


class ConstraintContrastiveLoss(nn.Module):
    def __init__(self,alpha=1.,M=20,useLabel=True):
        super(ConstraintContrastiveLoss, self).__init__()
        self.upper = alpha
        self.M = M
        self.useLabel = useLabel

    def forward(self, x, targets, x_supervise, label):
        x_sq = torch.sum(x.pow(2), dim=1).view(-1, 1)
        dissim_mat = F.relu(x_sq + x_sq.t() - 2. * x @ x.t()).sqrt()

        cluster_targets = self.computeMatch(targets)
        v1 = cluster_targets * dissim_mat.pow(2)[:,:,None]
        v2 = (cluster_targets-1.) * F.relu(self.upper - dissim_mat).pow(2)[:,:,None]
        loss = torch.mean(v1-v2,dim=[0,1])
        if self.useLabel:
            truth = (label.view(-1,1) == label.view(1,-1)).type(torch.float)

            x_sq_supervise = torch.sum(x_supervise.pow(2),dim=1).view(-1,1)
            dissim_mat_supervise = F.relu(x_sq_supervise + x_sq_supervise.t() - 2. * x_supervise @ x_supervise.t()).sqrt()
            v1 = truth * dissim_mat_supervise.pow(2)
            v2 = (truth-1.) * F.relu(self.upper - dissim_mat_supervise).pow(2)
            loss_label = torch.mean(v1-v2)
            m = torch.ones(targets.size(1)+1,dtype=x.dtype,device=x.device)
            m[-1] = self.M
            m = F.softmax(m,dim=-1)
            return m[:-1] @ loss + m[-1] * loss_label
        else:
            return loss.mean()

    def computeMatch(self,v):
        v1 = v.expand(v.shape[0], v.shape[0], v.shape[1])
        v2 = v1.transpose(0, 1)
        return (v1 == v2).type(torch.float)


def run(data_dic,param,seed):
    np.random.seed(seed)
    vis,prop = code4Relabel(data_dic['label_type'],data_dic['tot_class'],data_dic['mod_class'],data_dic['prop'],seed)
    testX, testY = getTest(data_dic['flag'])
    trainX, trainY, trainSeeny = getTrain(data_dic['flag'],vis,prop)
    cluster_label = getClusterLabel(data_dic['flag'],data_dic['selected_clusters'])

    posIdx = trainSeeny != -1
    trainX_supervise,trainY_supervise,cluster_label_supervise = trainX[posIdx],trainY[posIdx],cluster_label[posIdx]
    train_ds_supervise = DatasetWithClusterLabel(trainX_supervise, trainY_supervise, cluster_label_supervise)
    train_ds_unsupervise = DatasetWithClusterLabel(trainX, None, cluster_label)
    trainX_torch = torch.from_numpy(trainX).to(device).type(torch.float)
    testX_torch = torch.from_numpy(testX).to(device).type(torch.float)

    trainloader_supervise, valid_idx = get_train_valid_loader(train_ds_supervise,
        param['batch_size_supervise'] if len(train_ds_supervise) > param['batch_size_supervise'] else len(train_ds_supervise),data_dic['seed'],data_dic['valid_prop'],True)
    trainloader_unsupervise = DataLoader(train_ds_unsupervise,
        param['batch_size_unsupervise'] if len(train_ds_unsupervise) > param['batch_size_unsupervise'] else len(train_ds_unsupervise), True)
    labeled_train_iter = iter(trainloader_supervise)
    validX = trainX_supervise[valid_idx]
    validY = trainY_supervise[valid_idx]
    validX_torch = torch.from_numpy(validX).to(device).type(torch.float)
    unlabeled_train_iter = iter(trainloader_unsupervise)


    best_cluster = None
    best_sih = None
    best_ami = None
    best_m = None
    best_test_ami, best_test_acc = None,None
    for M in param['M']:
        fare = FARE(param['latent_dim'],data_dic['input_dim'],M,param['latent_dim'],param['use_label']).to(device)
        optimizer = Adam(fare.parameters(),lr=param['lr'],weight_decay=param['weight_decay'])

        for epoch in range(param['num_epoch']):
            x_supervise, label, _ = getIter(labeled_train_iter, trainloader_supervise)
            x, _, cluster_label = getIter(unlabeled_train_iter, trainloader_unsupervise)
            x_supervise = x_supervise.to(device).type(torch.float)
            x = x.to(device).type(torch.float)
            label = label.to(device).type(torch.long)
            cluster_label = cluster_label.to(device).type(torch.long)
            loss = fare(x, cluster_label, x_supervise, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'M:{M} training finished. Now try to find best K:')
        # now try different number of clusters
        with torch.no_grad():
            train_emb =fare.embedding(trainX_torch).cpu().numpy()
            test_emb = fare.embedding(testX_torch).cpu().numpy()
            test_ami, test_acc, sih, k = criteria_n_cluster(train_emb,test_emb,testY,param['K'])
            print(f'test_ami:{test_ami}')

            # ami is obtained from the validation set
            valid_emb = fare.embedding(validX_torch).cpu().numpy()
            valid_ami = compute_ami(train_emb,valid_emb,validY,k)

            if best_m is None:
                best_cluster = k
                best_sih = sih
                best_ami,best_test_ami = valid_ami,test_ami
                best_test_acc = test_acc
                best_m = M
                print(f'\t\tbest validation performance: ami:{valid_ami}; # of clusters: {k}')
            elif best_ami < valid_ami:
                best_cluster = k
                best_sih = sih
                best_ami,best_test_ami = valid_ami,test_ami
                best_test_acc = test_acc
                best_m = M
                print(f'\t\tbest validation performance: ami:{valid_ami}; # of clusters: {k}')
    print(f'best performance: ami:{best_test_ami}, acc:{best_test_acc}, sih:{best_sih}; # of clusters: {best_cluster}; best M:{best_m}')
    return best_test_ami, best_test_acc, best_sih, best_cluster, best_m


data_dic = {
    'flag':'malware',
    'label_type':'cg',
    'tot_class':6,
    'mod_class':2,
    'prop':0.01,
    'selected_clusters':-1, # select all
    'input_dim':100,
    'seed':101,
    'valid_prop':0.2,
}

param = {
    'latent_dim':32,
    'batch_size_unsupervise':128,
    'batch_size_supervise':32,
    'num_epoch':1000,
    'M':[1,3,5,7,10],
    'lr':1e-3,
    'weight_decay':1e-2,
    'use_label':True,
    'K':np.arange(4,10)
}


ami_avg = MovingAverage()
acc_avg = MovingAverage()
k_avg = MovingAverage()
M_avg = MovingAverage()
for seed in range(5):
    ami, acc, sih, k, m = run(data_dic,param,seed)
    ami_avg.add(ami)
    acc_avg.add(acc)
    k_avg.add(k)
    M_avg.add(m)
print(f'average ami:{ami_avg.get()}')
print(f'average acc:{acc_avg.get()}')
print(f'average K:{k_avg.get()}')
print(f'average M:{M_avg.get()}')
