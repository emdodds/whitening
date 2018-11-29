"""
PCA object that can be fit to and transform data. 
Written by Jesse Livezey.
Block functionality added by Eric Dodds.
"""

import numpy as np


class PCA:

    def __init__(self, dim = None, whiten = False, eps= 1e-8):
        """PCA object that can be fit to and transform data.

        Args:
            dim: Dimensionality to reduce to.
            whiten: Flag that tell pca to whiten the data before return. Default is False
            eps: Smallest allowed singular value
        """
        self.dim = dim
        self.whiten = whiten
        self.ready = False
        self.eps = eps

    def fit(self, data, row_col='r', blocks=-1):
        """Learns a basis for PCA

        Args:
            data: Data to do pca on.
            row_col: Flag that specifies how data is formatted.
            blocks : Size of blocks to break data into. By default, all one block

        Raises:
           ValueError: row_col flag not understood.
        """
        if row_col == 'c':
            data = data.T[:]
        elif row_col == 'r':
            pass
        else:
            raise ValueError('Malformed row_col flag.')
        self.nsamples = data.shape[0]
        if self.dim is None:
            self.dim = data.shape[1]
        else:
            assert self.dim <= data.shape[1]
    # Mean of each row
        self.mean_vec = data.mean(0)
    # Subtract mean
        center_vecs = data-self.mean_vec[np.newaxis,:]
    # Compute SVD
        if blocks > 0:
            self.sValues, v = self._block_fit(center_vecs, blocks)
        else:  
            if center_vecs.shape[0] > center_vecs.shape[1]:
                full_matrices = 0
            else:
                full_matrices = 1
    
            _, self.sValues, v = np.linalg.svd(center_vecs,
                                               full_matrices=full_matrices,
                                               compute_uv=1)
        idx = np.argsort(self.sValues)
        self.sValues = self.sValues[idx][::-1]
        self.eVectors = v[idx][::-1]
        self.ready = True
        
    def _block_fit(self, vecs, blocks):
        """
        Learns a basis for PCA using eigenvalue decomposition of the covariance
        matrix, which is computed from blocks of data. This may be needed when 
        using a lot of data, for memory reasons.
        
        Args:
            vecs: centered data
            blocks: number of data vectors per block
        """
        data_dim = vecs.shape[1]
        cov = np.zeros([data_dim, data_dim])
        nblocks = int(np.ceil(self.nsamples / blocks))
        for ind in range(nblocks):
            X = vecs[blocks*ind:blocks*(ind+1)]
            cov += X.T.dot(X)
        eigvals, eigvecs = np.linalg.eigh(cov)
        if not np.all(eigvals>0):
            print('Found a negative eigenvalue of covariance matrix, most negative value ' ,np.min(eigvals))
            eigvals = np.abs(eigvals)
        return np.sqrt(eigvals), eigvecs.T

    def fit_transform(self, data, row_col='r', dim=None, whiten=False, eps=1e-8,
                      blocks=-1):
        """Learns a basis for PCA and projects data onto it

        Args:
            data: Data to do pca on.
            row_col: Flag that specifies how data is formatted.
            dim: Dimensionality to reduce to.
            whiten: Flag that tell pca to whiten the data before return. Default is False
            eps: Smallest allowed singular value
            blocks: Size of blocks to break data into. By default, all one block

        Returns:
            reduced: Data with rediced dimensions.

        Raises:
           ValueError: rowColum flag not understood.
        """

        self.fit(data, row_col=row_col)
        return self.transform(data, row_col=row_col, dim=dim,
                              whiten=whiten, eps=eps, blocks=blocks)

    def transform(self, data, row_col='r', dim=None, whiten=None, eps=None, blocks=-1):
        """Projects vectors onto preexisting PCA basis and reduced dimensionality to dim.

        Args:
            data: Data to do pca on.
            row_col: Flag that specifies how data is formatted.
            dim: Dimensionality to reduce to.
            whiten: Flag that tells pca to whiten the data before return. Default is False
            eps: Smallest allowed singular value

        Returns:
            reduced_dim: Data with rediced dimensions.

        Raises:
            ValueError: rowColum flag not understood.
        """
        dim = dim or self.dim
        whiten = whiten or self.whiten
        eps = eps or self.eps
        if not self.ready:
            raise Exception('PCA model not yet fit with data')
        if row_col == 'c':
            data = data.T[:]
        elif row_col == 'r':
            pass
        else:
            raise ValueError('Malformed row_col flag.')

    #Subtract mean
        center_vecs = data-self.mean_vec[np.newaxis,:]
    #Project onto reduced number of eigenvectors.
        reduced_dim = center_vecs.dot(self.eVectors[:dim].T)
    #Whiten data if applicable
        if whiten:
            wm = np.diag(1./np.maximum(self.sValues, eps))
            reduced_dim = reduced_dim.dot(wm[:dim,:dim])
    #Transpose back to original
        if row_col == 'c':
            reduced_dim = reduced_dim.T
        return reduced_dim

    def inverse_transform(self, data, row_col='r', dim=None, whiten=None, eps=None):
        """Takes vectors from reduced dimensionality basis and returns them to full dimensionality basis.

        Args:
            data: Data to do pca on.
            row_col: Flag that specifies how data is formatted.
            dim: Dimensionality to reduce to.
            whiten: Flag that tell pca to whiten the data before return. Default is False
            eps: Smallest allowed singular value

        Returns:
            fullDim: Vectors in full dimensionality.

        Raises:
           ValueError: row_col flag not understood.
        """
        dim = dim or self.dim
        whiten = whiten or self.whiten
        eps = eps or self.eps
        if not self.ready:
            raise Exception('PCA model not yet fit with data')
        if row_col == 'c':
            full_data = data.T[:]
        elif row_col == 'r':
            full_data = data[:]
        else:
            raise ValueError('Malformed row_col flag.')

        if len(full_data.shape) < 2:
            full_data = full_data[np.newaxis,:] # a single vector has been input
        
        cur_dim = full_data.shape[1]
        if cur_dim != self.dim:
            raise ValueError('Data dimension is different than expected')
        if whiten:
            iwm = np.diag(np.maximum(self.sValues, eps))[:cur_dim,:cur_dim]
            full_data = full_data.dot(iwm)
        full_data = full_data.dot(self.eVectors[:cur_dim])
        full_data += self.mean_vec[np.newaxis,:]
        if row_col == 'c':
            full_data = full_data.T
        return full_data

    def transform_zca(self, data, row_col='r', dim=None, whiten=None, eps=None):
        """Projects vectors onto preexisting PCA basis 
        and reduced dimensionality to dim. Reproject back
        into data space.

        Args:
            data: Data to do zca on.
            row_col: Flag that specifies how data is formatted.
            dim: Dimensionality to reduce to.
            whiten: Flag that tell pca to whiten the data before return. Default is False
            eps: Smallest allowed singular value

        Returns:
            full_dim: Data with reduced dimensions.

        Raises:
            ValueError: rowColum flag not understood.
        """
        dim = dim or self.dim
        whiten = whiten or self.whiten
        eps = eps or self.eps
        if not self.ready:
            raise Exception('PCA model not yet fit with data')
        if row_col == 'c':
            data = data.T[:]
        elif row_col == 'r':
            pass
        else:
            raise ValueError('Malformed row_col flag.')

    #Subtract mean
        center_vecs = data-self.mean_vec[np.newaxis,:]
    #Project onto reduced number of eigenvectors.
        reduced_dim = center_vecs.dot(self.eVectors[:dim].T)

    #Whiten data if applicable
        if whiten:
            wm = np.diag(1./np.maximum(self.sValues, eps))
            reduced_dim = reduced_dim.dot(wm[:dim,:dim])
    #Project back to original space
        full_dim = reduced_dim.dot(self.eVectors[:dim])
    #Transpose back to original
        if row_col == 'c':
            full_dim = full_dim.T
        return full_dim
    
    def inverse_transform_zca(self, data, row_col='r', dim=None, whiten=None, eps=None):
        """Takes vectors from reduced dimensionality basis and returns them to full dimensionality basis.

        Args:
            data: Data to do zca on.
            row_col: Flag that specifies how data is formatted.
            dim: Dimensionality to reduce to.
            whiten: Flag that tell pca to whiten the data before return. Default is False
            eps: Smallest allowed singular value

        Returns:
            full_data: Vectors in full dimensionality.

        Raises:
           ValueError: row_col flag not understood.
        """
        dim = dim or self.dim
        whiten = whiten or self.whiten
        eps = eps or self.eps
        if not self.ready:
            raise Exception('PCA model not yet fit with data')
        if row_col == 'c':
            full_data = data.T[:]
        elif row_col == 'r':
            full_data = data[:]
        else:
            raise ValueError('Malformed row_col flag.')

        full_data = full_data.dot(self.eVectors[:dim].T)
        if whiten:
            iwm = np.diag(np.maximum(self.sValues, eps))[:dim,:dim]
            full_data = full_data.dot(iwm)
        full_data = full_data.dot(self.eVectors[:dim])
        full_data += self.mean_vec[np.newaxis,:]
        if row_col == 'c':
            full_data = full_data.T
        return full_data

    def explained_variance(self, dim = None):
        """Returns the fraction of the variance in the data to which the model was fit
        that is explained by the components retained."""
        if not self.ready:
            raise Exception('PCA model not yet fit with data')
        dim = dim or self.dim
        return np.sum(self.sValues[:dim]**2)/np.sum(self.sValues**2)