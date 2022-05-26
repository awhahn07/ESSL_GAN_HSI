from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sp

import os
import requests


class IP_DS(object):
    def __init__(self):

        '''Get Data from source or local if it exists'''
        data_url = 'http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat'
        label_url = 'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'

        data_set_path = 'Indian_pines_corrected.mat'
        label_path = 'Indian_pines_gt.mat'

        if data_set_path in os.listdir():
            pass
        else:
            r = requests.get(data_url)
            open(data_set_path, 'wb').write(r.content)

        if label_path in os.listdir():
            pass
        else:
            r = requests.get(label_url)
            open(label_path, 'wb').write(r.content)

        '''Load data and cast to float'''
        data = sp.loadmat(data_set_path)
        data = data['indian_pines_corrected'].astype('float32')

        lab = sp.loadmat('Indian_pines_gt.mat')['indian_pines_gt']

        '''Reshape Dataset'''
        self.data = np.reshape(
            data, [data.shape[0] * data.shape[1], data.shape[2]])

        self.lab = np.reshape(
            lab, [lab.shape[0] * lab.shape[1], 1]).astype('int64')

    def tensorfy(self, data):
        return np.reshape(data, [data.shape[0], data.shape[1], 1])

    def raw(self):
        return self.tensorfy(self.data), self.lab

    def scaled(self):
        scaler = make_pipeline(
            StandardScaler(), MinMaxScaler(feature_range=(-1, 1)))
        scaler.fit(self.data, self.lab)
        data_scaled = scaler.transform(self.data)
        return self.tensorfy(data_scaled), self.lab

    def PCA_n(self, n_components=None):
        pca_pipe = make_pipeline(
            StandardScaler(), PCA(n_components=n_components))
        pca_pipe.fit(self.data, self.lab)
        data_transformed = pca_pipe.transform(self.data)
        return self.tensorfy(data_transformed), self.lab

    def remove_null(self):
        Data, Lab = self.scaled()
        idx = [i for i, lab in enumerate(Lab) if lab != 0]
        Data = Data[idx]
        Lab = Lab[idx] - 1
        return Data, Lab

    def segment(self, dtype=None, n_components=None):
        if dtype == 'PCA':
            Data, Lab = self.PCA_n(n_components)
        elif dtype == 'raw':
            Data, Lab = self.raw()
        elif dtype == 'null':
            Data, Lab = self.remove_null()
        else:
            Data, Lab = self.scaled()

        Data = np.squeeze(Data)
        Lab = np.squeeze(Lab)
        lab_index = []
        classes = max(self.lab.squeeze()) + 1

        for j in range(classes):
            ind = [i for i, x in enumerate(Lab) if x == j]
            lab_index.append(ind)

        Segment = [Data[lab_index[z]] for z in range(classes)]
        if dtype == 'null':
            Segment.pop(-1)
        return Segment, lab_index

    def plot_data(self, dtype=None, n_components=None, size=20):
        Data, _ = self.segment(dtype=dtype, n_components=n_components)

        fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)
        fig.set_size_inches(10, 15)
        Data = np.squeeze(Data)

        print(len(Data))
        for i, axes in enumerate(ax.flat):
            if i < len(Data):
                idx = np.random.randint(
                    low=0, high=Data[i].shape[0], size=size)
                p = Data[i][idx, :]
                axes.plot(p.transpose())
                axes.set_title('Spectrum {}'.format(i + 1))
            axes.axis('on')
            axes.grid(True)
        return fig

    def get_weights(self, dtype='null'):
        Data, _ = self.segment(dtype=dtype)
        ls = [len(Data[i]) for i in range(len(Data))]
        mean = np.mean(ls)
        weights = [mean / ls[i] if mean / ls[i] > 1
                   else 1 for i in range(len(Data))]
        return weights
