import torch
import torch.nn as nn
import torch.optim as optim

from .struct_funcs import *
from .struct_pca_loss import *

lat_init_scale = 1e-10

class causal_linear_pca(nn.Module):
	def __init__(self, expr_mat, ndim_lat, 
			cc_mat, 
			size_factor, 
			batch_mat, 
			model='poisson',
			opt=1):
		super().__init__()
		
		# save size parameters
		self.ndim_lat = ndim_lat
		self.n_sample = expr_mat.shape[0]
		self.n_gene = expr_mat.shape[1]
		self.n_batch = batch_mat.shape[1]
	
		# linear pred function
		lat_pred_func = lat_pred_linear

		# initialize loss components
		self.expr_loss = expr_loss(ndim_lat, self.n_gene, self.n_batch, lat_pred_func, model)
		self.size_factor_loss = size_factor_loss(ndim_lat, self.n_batch, lat_pred_func)
		self.cell_cycle_loss = cell_cycle_loss(ndim_lat, self.n_batch, lat_pred_func)

		# initialize latent coordinates
		self.lat_coord = nn.Parameter(torch.randn((self.n_sample, self.ndim_lat)) * lat_init_scale)

		# opt
		self.opt = opt

		# save data
		self.expr_mat = expr_mat
		self.cc_mat = cc_mat
		self.size_factor = size_factor
		self.batch_mat = batch_mat
	
	def forward(self, expr_mat, cc_mat, size_factor, batch_mat):
		expr_loss = self.expr_loss(self.lat_coord, expr_mat, cc_mat, size_factor, batch_mat)
		size_factor_loss = self.size_factor_loss(self.lat_coord, cc_mat, size_factor, batch_mat)
		cell_cycle_loss = self.cell_cycle_loss(self.lat_coord, cc_mat, batch_mat)

		return expr_loss + self.opt * (- self.n_gene * size_factor_loss - self.n_gene * cell_cycle_loss)
	
	# custom backward someday def backward():

def fit_pca_linear(expr_mat, ndim_lat,
		cc_mat=None,
		size_factor=None,
		batch_mat=None,
		model='poisson',
		device=torch.device('cpu')
		):

	# n_sample
	opt = 1
	n_sample = expr_mat.shape[0]
	
	# supply zero to unspecified inputs
	if cc_mat == None:
		cc_mat = torch.zeros(n_sample, 2, device=device)
	if size_factor == None:
		size_factor = torch.ones(n_sample, device=device)
	if batch_mat == None:
		batch_mat = torch.zeros(n_sample, 1, device=device)

	# invoke linear pca
	model = causal_linear_pca(expr_mat, ndim_lat, cc_mat, size_factor, batch_mat, model, opt)

	# load data to device
	model.to(device)
	expr_mat = expr_mat.to(device)
	cc_mat = cc_mat.to(device)
	size_factor = size_factor.to(device)
	batch_mat = batch_mat.to(device)

	# init batch params
	batch_mat_agg = torch.ones((n_sample,1), device=device)
	A = batch_mat_agg.T @ batch_mat_agg
	B = batch_mat_agg.T @ (expr_mat / size_factor[:,None])
	alpha, _ = torch.solve(B, A)
	model.expr_loss.logmean_layer.const.data = alpha.log()
	model.size_factor_loss.logsf_mean_layer.const.data[0,0] = size_factor.log().mean()

	optim_all = optim.Adam(model.parameters(), lr=0.05)
	optim_all.zero_grad()
	loss = -model(expr_mat, cc_mat, size_factor, batch_mat)
	loss.backward()
	for itr in range(2000):
		if itr%40 == 0:
			print('step %s: loss %.2f' % (itr, loss))
		optim_all.step()
		optim_all.zero_grad()
		loss_old = loss
		loss = -model(expr_mat, cc_mat, size_factor, batch_mat)
		if torch.abs((loss - loss_old)/loss_old) < 1e-6:
			break
		loss.backward()

	F, D, V = torch.svd(model.expr_loss.logmean_layer.lat_coef)
	U = (model.lat_coord @ F @ torch.diag(D))

	return U
