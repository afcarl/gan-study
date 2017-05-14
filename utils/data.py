'''
data.py

Data loading and preprocessing auxiliary methods.
'''

from torch.utils.data import Dataset

class ConcDataset(Dataset):
    '''
    Combines multiple datasets in a single dataset.
    See https://discuss.pytorch.org/t/combine-concat-dataset-instances/1184.
    '''

    def __init__(self, datasets):
        '''
        Initializes ConcDataset.

        @param datasets datasets list.
        '''

        # Initializing
        self.datasets = datasets
        self.samples = [s for d in datasets for s in d]
        self.len = len(self.samples)

    def __getitem__(self, index):
        '''
        Returns sample with given index.

        @param index sample index.
        @return sample.
        '''
        return self.samples[index]

    def __len__(self):
        '''
        Dataset sample size.
        '''
        return self.len
