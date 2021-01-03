import numpy as np
import torch
from scipy import io
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

def getTest(flag):
    dataDict = io.loadmat(f'data/{flag}_test.mat')
    testX = dataDict['feature']
    testy = dataDict['full_label'].reshape(-1)
    return testX,testy


def getTrain(flag,changeVisual,changeProp):
    dataDict = io.loadmat(f'data/{flag}_train.mat')
    trainX = dataDict['feature']
    trainy = dataDict['full_label'].reshape(-1)
    if changeVisual:
        trainSeeny = dataMask(trainy,True,changeVisual,changeProp)
    return trainX,trainy,trainSeeny


def dataMask(y,isTrain,visualLabel,prop,seed=101):
    # re-format labels
    np.random.seed(seed)
    trainSeenLabel = []
    if isTrain:
        for i, label in enumerate(y):
            inTest = False
            for j, visuals in enumerate(visualLabel):
                if label in visuals:
                    inTest = True
                    u = np.random.uniform()
                    if u < prop[j]:  # this label is observed
                        trainSeenLabel.append(visuals[0])
                    else:
                        trainSeenLabel.append(-1)
            if not inTest:
                trainSeenLabel.append(-1)
    else:
        for i, label in enumerate(y):
            inTest = False
            for j, visuals in enumerate(visualLabel):
                if label in visuals:
                    inTest = True
                    trainSeenLabel.append(visuals[0])
            if not inTest:
                trainSeenLabel.append(-1)
    return np.array(trainSeenLabel)


def getClusterLabel(flag,num_select=-1):
    cluster_label = np.load('./data/'+flag+'.npy')
    cols = np.arange(cluster_label.shape[1])
    if num_select>0:
        selected_cols = np.random.choice(cols, size=num_select, replace=False)
    else:
        selected_cols = cols
    cluster_label = cluster_label[:, selected_cols]
    return cluster_label


class DatasetWithClusterLabel(Dataset):
    def __init__(self, imgs, true_labels, cluster_labels):
        self.imgs = imgs
        self.true_labels = true_labels
        self.cluster_labels = cluster_labels

    def __getitem__(self, idx):
        if self.true_labels is None:
            return self.imgs[idx],-1,self.cluster_labels[idx]
        else:
            return self.imgs[idx], self.true_labels[idx], self.cluster_labels[idx]

    def __len__(self):
        return len(self.imgs)


def get_train_valid_loader(train_ds,
                           batch_size,
                           random_seed,
                           valid_size=0.2,
                           shuffle=True):
    num_train = len(train_ds)
    indices = np.arange(num_train)
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size if batch_size < len(train_idx) else len(train_idx), sampler=train_sampler)
    return train_loader, valid_idx


def code4Relabel(labelType, tot_class, mod_class, prop,seed):
    np.random.seed(seed)
    if labelType == 'mc':
        print(f'Missing class setting. Total class:{tot_class}, # of missing:{mod_class}')
        seen_class = tot_class - mod_class
        choice = np.random.choice(np.arange(tot_class), seen_class, False)
        changeVisual = [[x] for x in choice]
        changeProp = [prop] * len(changeVisual)

    elif labelType == 'cg':
        print(f'Coarse-grained label setting. Total class:{tot_class}, # of grouping:{mod_class}')
        choice = np.arange(tot_class)
        np.random.shuffle(choice)
        changeVisual = [choice[:mod_class]]
        for i in range(mod_class,tot_class):
            changeVisual.append([choice[i]])
        changeProp = [prop] * len(changeVisual)
    else:
        raise Exception("Unknown label type. Please use either 'mc' (missing class) or 'cg' (coarse-grained label).")
    return changeVisual,changeProp


