import torch
import torch.nn as nn

from .struct_funcs import *
from .utils import *

class expr_loss(nn.Module):
	def __init__(self, ndim_lat, n_gene, n_batch, lat_pred_func, model='poisson'):
		super(expr_loss, self).__init__()

		# stores expr related parameters
		self.logmean_layer = expr_logmean_linear(ndim_lat, n_gene, n_batch, lat_pred_func)	

		# select prob model
		if model == 'poisson':
			self.pmf = poisson_logpmf()
		elif model =='nb':
			self.pmf = nb_logpmf(n_gene)
		else:
			# error messeage implementation
			pass
	
	def forward(self, lat_coord, expr_mat, cc_mat, size_factor, batch_mat):
		logmean = self.logmean_layer(lat_coord, cc_mat, size_factor, batch_mat)
		probs = self.pmf(expr_mat, logmean)	
		
		return probs.sum(dim=1).mean()

class size_factor_loss(nn.Module):
	def __init__(self, ndim_lat, n_batch, lat_pred_func):
		super(size_factor_loss, self).__init__()

		# stores size factor related parameters
		self.logsf_mean_layer = log_size_factor_mean_linear(ndim_lat, n_batch, lat_pred_func)
		# MSE loss
		self.mseloss = nn.MSELoss()
	
	def forward(self, lat_coord, cc_mat, size_factor, batch_mat):
		logsf_mean = self.logsf_mean_layer(lat_coord, cc_mat, batch_mat)

		return self.mseloss(logsf_mean, size_factor.log().unsqueeze(-1))

class cell_cycle_loss(nn.Module):
	def __init__(self, ndim_lat, n_batch, lat_pred_func):
		super(cell_cycle_loss, self).__init__()

		# stores cell cycle related parameters
		self.cc_mean_layer = cell_cycle_mean_linear(ndim_lat, n_batch, lat_pred_func)	
		# MSE loss
		self.mseloss = nn.MSELoss()

	def forward(self, lat_coord, cc_mat, batch_mat):
		cc_mean = self.cc_mean_layer(lat_coord, batch_mat)
		
		return self.mseloss(cc_mean, cc_mat)
