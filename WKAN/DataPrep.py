from.Module import *

import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch import tensor as Tensor
import numpy as np
import scipy
from scipy.sparse import issparse

class wrap_gene_location(TensorDataset):
    # """Dataset wrapping labeled (cluster label) data tensors with cluster information.
    # Used in data prediction models
    # Each sample will be retrieved by indexing tensors along the first
    # dimension.
    #
    # Arguments:
    # 	datainput (numpy array): contains sample data.
    # """
    def __init__(self, datainput, label):
        self.data_tensor = torch.from_numpy(datainput).float()
        cord = label.to_numpy().astype('float32')
        cordx = cord[:, 0]
        cordy = cord[:, 1]
        self.xmin = cordx.min() - 1
        self.ymin = cordy.min() - 1
        self.xmax = cordx.max() + 1
        self.ymax = cordy.max() + 1
        self.cordx_norm = (cordx - self.xmin) / (self.xmax - self.xmin)
        self.cordy_norm = (cordy - self.ymin) / (self.ymax - self.ymin)
        self.imagedimension = self.data_tensor.shape

    def __getitem__(self, index):
        indexsample = index // self.imagedimension[2]
        indexspot = index % self.imagedimension[2]
        geneseq = self.data_tensor[indexsample, :, indexspot]
        cordinates = torch.tensor([self.cordx_norm[indexspot], self.cordy_norm[indexspot]])
        return geneseq, cordinates

    def __len__(self):
        return self.imagedimension[0] * self.imagedimension[2]




class wrap_gene_layer(TensorDataset):
	"""Dataset wrapping labeled (cluster label) data tensors with cluster information.
	Used in data prediction models
	Each sample will be retrieved by indexing tensors along the first
	dimension.

	Arguments:
		datainput (numpy array): contains sample data.
		layer (boolean): T if layer information is contained
		layerkey: the keyword for layer. Default is "Layer"
	"""
	def __init__(self, datainput, label, layerkey = "layer"):
		self.data_tensor = torch.from_numpy(datainput).float()
		getlayer = label[layerkey].to_numpy()
		self.layer = getlayer.astype('float32')
		self.layersunq = np.sort(np.unique(self.layer))
		self.nlayers = len(self.layersunq)
		self.imagedimension = self.data_tensor.shape
	def __getitem__(self, index):
		indexsample = index // self.imagedimension[2]
		indexspot = index % self.imagedimension[2]
		geneseq = self.data_tensor[indexsample,:,indexspot]
		layeri = int(self.layer[indexspot]) - 1
		layerv = np.zeros(self.nlayers-1)
		layerv[:layeri] = 1
		return geneseq, layerv
	def __len__(self):
		return self.imagedimension[0] * self.imagedimension[2]




def get_zscore (adata, mean = None, sd = None ):
	genotypedata = (adata.X.A if issparse(adata.X) else adata.X)
	if mean is None:
		genemean = np.mean(genotypedata, axis =0)
		genesd = np.std(genotypedata, axis = 0)
	else:
		genemean = mean
		genesd = sd
	try:
		if adata.standardize is not True:
				datatransform = (genotypedata - genemean) / genesd
				adata.X = datatransform
				adata.genemean = genemean
				adata.genesd = genesd
				adata.standardize = True
		else:
			print("Data has already been z-scored")
	except AttributeError:
		datatransform = (genotypedata - genemean) / genesd
		adata.X = datatransform
		adata.genemean = genemean
		adata.genesd = genesd
		adata.standardize = True