# FARE: Enabling Fine-grained Attack Categorization under Low-quality Labeled Data

Supervised machine learning classifiers have been widely used for attack detection, but their training requires abundant high-quality labels. Unfortunately, high-quality labels are difficult to obtain in practice due to the high cost of data labeling and the constant evolution of attackers. Without such labels, it is challenging to train and deploy targeted countermeasures.

In this paper, we propose *FARE*, a clustering method to enable fine-grained attack categorization under low-quality labels. We focus on two common issues in data labels: 1) missing labels for certain attack classes or families; and 2) only having coarse-grained labels available for different attack types. The core idea of *FARE* is to take full advantage of the limited labels while using the underlying data distribution to consolidate the low-quality labels. We design an *ensemble model* to fuse the results of multiple unsupervised learning algorithms with the given labels to mitigate the negative impact of missing classes and coarse-grained labels. We then train an input transformation network to map the input data into a low-dimensional latent space for fine-grained clustering. 

For more details, please refer to our [paper](./ndss21.pdf).

## Notes

We only upload a subset of our data due to file size limit. The main code file is [FARE.py](./FARE.py). To run the code, simply type `python FARE.py`.

Implementation of base clustering method DEC and other deep learning based clustering and anomaly detection methods: https://github.com/Henrygwb/UnsupervisedLearing.

## Data format

In the `data` folder, we split the input data into three files. A file containing all training data (`malware_train.mat`), a file containing all testing data (`malware_test.mat`) and a file containing the clustering results output from the neighborhood models (`malware.npy`). The structures of the data file are as follows:

* `malware_train.mat` is exported using `scipy.io.savemat` function. It has two fields, namely `{feature, full_label}`. `feature` is the feature matrix associated to the samples and `full_label` is the label set. Please check [readData.py](./readData.py) for more details.
* `malware_test.mat`: same as `malware_train.mat`
* `malware.npy` is exported using `numpy.save` function. It is a matrix with dimension $M\times N$, where $N$ is the sample size and $M$ is the number of neighborhood models. The entry $v_{nm}$ represents the cluster index of sample $n$ in neighborhood model $m$.

## Citation

```
@inproceedings{liang2021fare,
  title={FARE: Enabling Fine-grained Attack Categorization under Low-quality Labeled Data},
  author={Liang, Junjie and Guo, Wenbo and Luo, Tongbo and Honavar, Vasant and Wang, Gang and Xing, Xinyu},
  booktitle={The Network and Distributed System Security Symposium 2021},
  year={2021}
}
```

If you have any question, please feel free to contact me [Junjie Liang](mailto:jul672@ist.psu.edu)

