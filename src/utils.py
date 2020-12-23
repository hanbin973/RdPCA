import torch
import torch.nn as nn

class poisson_logpmf(nn.Module):
	def __init__(self):
		super(poisson_logpmf, self).__init__()

	def forward(self, count, logmean):
		return count * logmean - logmean.exp()

class nb_logpmf(nn.Module):
	def __init__(self, n_gene):
		super(nb_logpmf, self).__init__()

		# initialize dispersion parameters
		self.disp_params = nn.Parameter(torch.ones(n_gene))
	
	def forward(self, count, logmean):
		# define nb log pmf
		pass
