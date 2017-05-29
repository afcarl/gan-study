'''
data.py

Data loading and preprocessing auxiliary methods.
'''

import os # OS resources
import numpy as np # numpy library
import pandas as pd # Pandas library (csv reader, etc...)

from sklearn.preprocessing import LabelBinarizer # one-hot encoder

import torch # Torch operations
from torch.utils.data import Dataset # Dataset class

from PIL import Image # Image handler
from PIL import ImageFile # To solve load problems

# Valid images extensions
IMG_EXTENSIONS = [
	'.jpg', '.JPG', '.jpeg', '.JPEG',
	'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
ImageFile.LOAD_TRUNCATED_IMAGES = True

def is_image_file(filename):
    '''
    Tests if current file is a valid image.

    @param filename input file name.
    @return True if file is a image, else False.
    '''
    return filename.endswith(tuple(IMG_EXTENSIONS))

def load_data(root, datafile):
    '''
    Loads dataset data.

    @param root images folder root path.
    @param datafile datafile file.
    @return Tuples with images paths and targets, and targets labels.
    '''

    # Opening file and extracting images list
    dataset = pd.read_csv(datafile, index_col='imgs')
    images = dataset.index.values.tolist()

    # Converting target columns to float values and saving target labels
    targets = dataset.values.astype(float).tolist()
    tlabels = dataset.columns.values.tolist()

    # Setting samples
    samples = []
    for image, target in zip(images, targets):

        # Testing for a valid image
        if is_image_file(image):

            # Converting targets
            target = torch.from_numpy(np.float64(target)).float()

            # Setting image sample
            path = os.path.join(root, image)
            item = (path, target)
            samples.append(item)

    # Return
    return samples, tlabels

def default_loader(path):
    '''
    Loads image on path.

    @param path image path.
    @return Image.
    '''
    return Image.open(path).convert('RGB')

class ToOneHot(object):
	'''
	One-hot encoder transform.
	'''

	def __init__(self, classes):
		'''
		Inits a ToOneHot instance.
		@param classes Class vector.
		'''

		# Set encoder
		self.encoder = LabelBinarizer()
		self.encoder = self.encoder.fit(classes)

	def __call__(self, tensor):
		'''
		Input call.
		@param input torch tensor.
		@return output tensor.
		'''

		# Convert tensor/int to numpy and computing labels
		if isinstance(tensor, ( int, long )):
			tensor = np.array([tensor])
		elif isinstance(tensor, torch.Tensor):
			tensor = tensor.numpy()
		labels = torch.from_numpy(self.encoder.transform(tensor)).float()
		labels = labels.squeeze()
		return labels

class DataCSV(Dataset):
    '''
    Loads a image database from a reference csv file.
    '''

    def __init__(self, root, datafile, transform=None, target_transform=None,\
    loader=default_loader):
        '''
        Inits a DataCSV instance.

        @param root Images root folder path.
        @param datafile Data file path.
        @param transform Images transforms.
        @param target_transform Labels transform.
        @param loader Images loader.
        '''

        # Loads data
        imgs, labels = load_data(root, datafile)
        if len(imgs) == 0:
            # Error
            raise(RuntimeError("Found 0 images in path: " + root + "\n"
                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        # Saving data
        self.root = root
        self.datafile = datafile
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        # Main data
        self.classes = labels
        self.imgs = imgs

    def __getitem__(self, index):
        '''
        Returns image and target values for a given index.

        @param index Input index.

        @return The image and its respective target.
        '''

        # Get images
        path, target = self.imgs[index]
        img = self.loader(path)

        # Transforming image
        if self.transform is not None:
            img = self.transform(img)

        # Transforming target labels
        if self.target_transform is not None:
            target = self.target_transform(target)

        # Return
        return img, target

    def __len__(self):
        '''
        Returns samples size.

        @return Current data number of samples.
        '''
        return len(self.imgs)


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
