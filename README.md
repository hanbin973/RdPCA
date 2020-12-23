# Read-depth aware Principal Component Analysis (RdPCA) for scRNA-seq data

Read-depth aware principal component analysis (RdPCA) is a extension of principal component analysis (RdPCA) that extracts biological signals from read-depth to improve dimension reduction in scRNA-seq data analysis.

## Basic usage
`fit_pca_linear` function in `model.py` returns the RdPCA coordinates.
```
fit_pca_linear(expr_mat, 
		ndim_lat,
		cc_mat=None,
		size_factor=None,
		batch_mat=None,
		model='poisson',
		device=torch.device('cpu')
		)
```
* `expr_mat`: `n_sample` x `n_gene` expression matrix
* `ndim_lat`: number of principal components
* `cc_mat`: `n_sample` x 2 cell cycle phase score matrix (e.g. Seurat cell cycle score)
* `size_factor`: `n_sample` vector of offset variables (e.g. scran size factors)
* `model`: currently only `poisson` is supported
* `device`: torch device (default: cpu)



