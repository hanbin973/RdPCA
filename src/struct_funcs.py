import torch
import torch.nn as nn

lat_init_scale = 1e-10

# this linear layer can be replaced by any functional form
class lat_pred_linear(nn.Module):
	def __init__(self):
		super(lat_pred_linear, self).__init__()
	
	def forward(self, lat_coord, lat_coef):
		return torch.matmul(lat_coord, lat_coef)

class expr_logmean_linear(nn.Module):
	def __init__(self, ndim_lat, n_gene, n_batch, lat_pred_func=lat_pred_linear):
		super(expr_logmean_linear, self).__init__()

		# latent coef: latent dimension x genes
		self.lat_coef = nn.Parameter(torch.randn((ndim_lat, n_gene)) * lat_init_scale)
		# cell cycle coef: (g2m+s) x genes
		self.cc_coef = nn.Parameter(torch.zeros((2, n_gene)))
		# constant coef: 1 x genes
		self.const = nn.Parameter(torch.zeros((1, n_gene)))
		# batch coef: batch x genes
		self.batch_coef = nn.Parameter(torch.zeros((n_batch, n_gene)))
		
		# the latent predictor function to be used
		self.lat_pred_func = lat_pred_func()

	def forward(self, lat_coord, cc_mat, size_factor, batch_mat):
		lat_pred = self.lat_pred_func(lat_coord, self.lat_coef)
		return lat_pred + \
				torch.matmul(cc_mat, self.cc_coef) + \
				size_factor.log()[:,None] + \
				self.const.expand(lat_pred.shape[0], -1) + \
				torch.matmul(batch_mat, self.batch_coef) 

class log_size_factor_mean_linear(nn.Module):
	def __init__(self, ndim_lat, n_batch, lat_pred_func=lat_pred_linear):
		super(log_size_factor_mean_linear, self).__init__()

		# latent coef: latent dimension x 1
		self.lat_coef = nn.Parameter(torch.randn((ndim_lat, 1)) * lat_init_scale)
		# cell cycle coef: (g2m+s) x 1
		self.cc_coef = nn.Parameter(torch.zeros((2, 1)))
		# constant coef: 1 x 1
		self.const = nn.Parameter(torch.zeros((1, 1)))
		# batch coef: batch x 1
		self.batch_coef = nn.Parameter(torch.zeros((n_batch, 1)))

		# the latent predictor function to be used
		self.lat_pred_func = lat_pred_func()

	def forward(self, lat_coord, cc_mat, batch_mat):
		lat_pred = self.lat_pred_func(lat_coord, self.lat_coef)
		return lat_pred + \
				torch.matmul(cc_mat, self.cc_coef) + \
				self.const.expand(lat_pred.shape[0], -1) + \
				torch.matmul(batch_mat, self.batch_coef)	

class cell_cycle_mean_linear(nn.Module):
	def __init__(self, ndim_lat, n_batch, lat_pred_func=lat_pred_linear):
		super(cell_cycle_mean_linear, self).__init__()
		
		# latent coef: latent dimension x 2
		self.lat_coef = nn.Parameter(torch.randn((ndim_lat, 2)) * lat_init_scale)
		# constant coef: 1 x 2
		self.const = nn.Parameter(torch.zeros((1, 2)))
		# batch coef: batch x 2
		self.batch_coef = nn.Parameter(torch.zeros((n_batch, 2)))

		# the latent predictor function to be used
		self.lat_pred_func = lat_pred_func()

	def forward(self, lat_coord, batch_mat):
		lat_pred = self.lat_pred_func(lat_coord, self.lat_coef)
		return lat_pred + \
				self.const.expand(lat_pred.shape[0], -1) + \
				torch.matmul(batch_mat, self.batch_coef)
