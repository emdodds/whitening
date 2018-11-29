# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 18:23:08 2015

@author: Eric Dodds
"""
import numpy as np
# try/except block gets around an issue on the cluster
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    print("Plotting unavailable.")


class StimSet(object):
    def __init__(self, data, stimshape, batch_size=None):
        """Notice that stimshape and the length of a datum may be different, since the
        data may be represented in a reduced form."""
        self.data = data
        self.stimshape = stimshape
        self.stimsize = np.prod(stimshape)
        self.nstims = data.shape[0]
        self.datasize = data.shape[1]
        self.batch_size = batch_size

    def rand_stim(self, batch_size=None):
        """Select random inputs. Return an array of batch_size columns,
        each of which is an input represented as a (column) vector. """
        batch_size = batch_size or self.batch_size
        veclength = np.prod(self.datasize)
        X = np.zeros((veclength, batch_size))
        for i in range(batch_size):
            which = np.random.randint(self.nstims)
            vec = self.data[which, ...]
            if len(vec.shape) > 1:
                vec = vec.reshape(self.stimsize)
            X[:, i] = vec
        return X

    @staticmethod
    def _stimarray(stims, stimshape, layout='sqrt'):
        """Returns an array of the stimuli reshaped to 2d and tiled."""
        length, height = stimshape
        assert length*height == stims.shape[1]
        buf = 1  # buffer pixels between stimuli
        nstim = stims.shape[0]

        # n and m are respectively the numbers of rows and columns of stimuli in the array
        n, m = StimSet._get_layout(nstim, length, height, layout)

        array = 0.5*np.ones((buf+n*(length+buf), buf+m*(height+buf)))
        k = 0

        for i in range(m):
            for j in range(n):
                if k < nstim:
                    normfactor = np.max(np.abs(stims[k, :]))
                    hstart = buf+i*(height+buf)
                    lstart = buf+j*(length+buf)
                    thestim = stims[k, :].reshape(length, height)/normfactor
                    array[lstart:lstart+length, hstart:hstart+height] = thestim

                k = k+1

        return array.T

    @staticmethod
    def _get_layout(nstim, length=None, height=None, layout='square'):
        """Get a number of rows n and columns m corresponding to a given
        layout type. Returns the argument if it's already a pair."""
        if layout == 'square':
            if np.floor(np.sqrt(nstim))**2 != nstim:
                n = int(np.ceil(np.sqrt(nstim/2.)))
                m = int(np.ceil(nstim/n))
            else:
                # M is a perfect square
                m = int(np.sqrt(nstim))
                n = m
        elif layout == 'sqrt':
            # if length != height, partly account for this so stimuli aren't so distorted. 
            # could remove the extra square root to fully accommodate
            n = int(np.sqrt(nstim*np.sqrt(height/length)))
            m = int(np.ceil(nstim/n))
        else:
            n, m = layout
        return n, m

    def stimarray(self, stims, stimshape=None, layout='sqrt'):
        stimshape = stimshape or self.stimshape
        return StimSet._stimarray(stims, stimshape, layout)

    def modspec(self, elem):
        """Compute the modulation power spectrum."""
        image = elem.reshape(self.stimshape)
        fourier = np.fft.rfft2(image)
        mid = int(fourier.shape[0]/2)
        power = np.abs(fourier)**2
        avgmag = np.array([(power[ii] + power[-ii])/2 for ii in range(mid)])
        return avgmag

    def stim_for_display(self, stim):
        return stim.reshape(self.stimshape)

    def tiledplot(self, stims, cmap='RdBu_r', layout='square',
                  aspect='auto', savestr=None):
        nstim = stims.shape[0]
        length, height = self.stimshape
        n, m = StimSet._get_layout(nstim, length, height, layout)
        gs = gridspec.GridSpec(m, n, wspace=0.0, hspace=0.0)
        for ii in range(m):
            for jj in range(n):
                ax = plt.subplot(gs[ii, jj])
                index = ii*n + jj
                if index < nstim:
                    image = self.stim_for_display(stims[index])
                    image /= np.max(np.abs(image))
                    ax.imshow(image.T,
                              cmap=cmap, origin='lower', aspect=aspect,
                              interpolation='nearest',
                              clim=(-1.0, 1.0))
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        if savestr is not None:
            plt.savefig(savestr, bbox_inches='tight')


class ImageSet(StimSet):
    """Container for image data. The 'stimuli' are patches drawn randomly from
    the set of images."""

    def __init__(self, data, stimshape=(16, 16), batch_size=None,
                 buffer=20, patchwisenorm=False):
        self.buffer = buffer
        self.patchwisenorm = patchwisenorm
        StimSet.__init__(self, data, stimshape, batch_size)
        self.datasize = np.prod(stimshape)  # size of a patch

    def rand_stim(self, stimshape=None, batch_size=None):
        """
        Select random patches from the image data. Returns data array of
        batch_size columns, each of which is an unrolled image patch of size
        prod(stimshape).
        """
        batch_size = batch_size or self.batch_size or 100
        length, height = stimshape or self.stimshape
        # extract subimages at random from images array to make data array X
        X = np.zeros((length*height, batch_size))
        for i in range(batch_size):
            which = np.random.randint(self.data.shape[-1])
            nrows, ncols = self.data[:, :, which].shape
            row = self.buffer + int(np.ceil((nrows-length-2*self.buffer)*np.random.rand()))
            col = self.buffer + int(np.ceil((nrows-height-2*self.buffer)*np.random.rand()))
            animage = self.data[row:row+length,
                                col:col+height,
                                which]
            animage = animage.reshape(self.stimsize)
            if self.patchwisenorm:
                # normalize image
                animage -= animage.mean()
                animage /= animage.std()
            X[:, i] = animage
        return X


class PCvecSet(StimSet):
    """Principal component vector representations of arbitrary data."""

    def __init__(self, data, stimshape, pca, batch_size=None):
        self.pca = pca
        self.datasize = data.shape[1]
        StimSet.__init__(self, data, stimshape, batch_size)

    def stimarray(self, stims, layout='sqrt'):
        reconst = self.pca.inverse_transform(stims)
        return StimSet.stimarray(self, reconst, self.stimshape, layout)

    def modspec(self, elem):
        return StimSet.modspec(self, self.pca.inverse_transform(elem))

    def stim_for_display(self, stim):
        return StimSet.stim_for_display(self, self.pca.inverse_transform(stim))


class SpectroPCSet(PCvecSet):
    """A PCvecSet with some extra functionality specifically for spectrograms."""
    def __init__(self, data, stimshape, pca, batch_size=None,
                 freqs=None, tbin_width=None):
        """
        Parameters:
        freqs : array_like of the frequencies sampled, in Hz
        tbin_width : time in ms separating centers of adjacent time bins
        """
        # spectrogram parameters default to those in Carlson, Ming, & DeWeese 2012
        self.tbin_width = tbin_width or 8
        self.freqs = freqs or np.logspace(2, np.log10(16000/4), 256)
        PCvecSet.__init__(self, data, stimshape, pca, batch_size)

    def show_stim(self, stim, cmap='RdBu_r', savestr=None, cbar=False):
        reshaped = self.stim_for_display(stim)
        tlength, nfreqs = self.stimshape
        vmax = np.max(reshaped)
        plt.imshow(reshaped.T, interpolation='nearest',
                   cmap=cmap, aspect='auto', origin='lower',
                   vmin=-vmax, vmax=vmax)
        plt.ylabel('Frequency')
        plt.xlabel('Time (ms, bin = '+str(self.tbin_width)+' ms)')
        middlef = str(int(self.freqs[int(nfreqs/2)]))
        middlet = str(int(self.tbin_width*(tlength+1)/2))
        endtime = (tlength + 2)*self.tbin_width
        plt.xticks([0, int(tlength/2)+1, tlength-1], ['0', middlet, endtime])
        plt.yticks([0, int(nfreqs/2), nfreqs-1],
                   [str(self.freqs[0])+' Hz', middlef+' Hz', str(int(self.freqs[-1]/1000)) + ' kHz'])
        if cbar:
            plt.colorbar()
        if savestr is not None:
            plt.savefig(savestr, bbox_inches='tight')
        plt.show()

    def show_set(self, stims, cmap='RdBu_r', layout=(4, 5), savestr=None):
        """
        Parameters:
        stims : (number of stim, flattened stim length) stimuli to plot
        layout : (number of rows, number of columns) per figure
        """
        tlength, nfreqs = self.stimshape
        per_figure = np.prod(layout)
        nstim = stims.shape[0]

        plt.figure()
        for ii in range(nstim):
            if ii % per_figure == 0 and ii > 0:
                if savestr is not None:
                    plt.tight_layout()
                    plt.subplots_adjust(wspace=.05, hspace=.05)
                    plt.savefig(savestr+str(int(ii/per_figure)), bbox_inches='tight')
                plt.figure()
            plt.subplot(layout[0], layout[1], (ii % per_figure)+1)
            plt.imshow(self.stim_for_display(stims[ii]).T, interpolation='nearest',
                       cmap=cmap, aspect='auto', origin='lower')
            if ii % per_figure == per_figure - layout[1]:
                # label axes for bottom left example
                plt.ylabel('Frequency')
                plt.xlabel('Time (ms, bin = '+str(self.tbin_width/2)+' ms)')
                middlef = str(int(self.freqs[len(self.freqs)/2]))
                middlet = str(int(self.tbin_width/2*(tlength+1)/2))
                endtime = str(int((tlength + 2)*self.tbin_width/2))
                plt.xticks([0, int(tlength/2)+1, tlength-1], ['0', middlet, endtime])
                plt.yticks([0, int(nfreqs/2), nfreqs-1],
                    [str(int(self.freqs[0]))+' Hz', middlef+' Hz', str(int(self.freqs[-1]/1000))+ ' kHz'])
            else:
                plt.gca().get_yaxis().set_visible(False)
                plt.gca().get_xaxis().set_visible(False)

        plt.tight_layout()
        plt.subplots_adjust(wspace=.05, hspace=.05)
        if savestr is not None:
            plt.savefig(savestr, bbox_inches='tight')
        plt.show()


class WaveformSet(StimSet):
    """1D signals, especially audio, of uniform length."""

    def tiledplot(self, stims):
        """Tiled plots of the given stumili. Zeroth index is over stimuli.
        Kind of slow, expect about 10s for 100 plots."""
        nstim = stims.shape[0]
        plotrows = int(np.sqrt(nstim))
        plotcols = int(np.ceil(nstim/plotrows))
        f, axes = plt.subplots(plotrows, plotcols, sharex=True, sharey=True)
        for ii in range(nstim):
            axes.flatten()[ii].plot(stims[ii])
        f.subplots_adjust(hspace=0, wspace=0)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        plt.setp([a.get_yticklabels() for a in f.axes[:-1]], visible=False)


class WaveformPCSet(PCvecSet, WaveformSet):
    """Specifically for PCA reps of waveforms, i.e., 1D time series."""

    def tiledplot(self, stims):
        WaveformSet.tiledplot(self, self.pca.inverse_transform(stims))


class ToySparseSet(StimSet):
    """Gaussian sources linearly mixed by laplacian coefficients,
    optiohally with isotropic gaussian noise added."""
    def __init__(self, dim=200, nsource=None, scale=1,
                 nonneg=False, nstims=300000, rng=None,
                 batch_size=100, white=False, noise=0,
                 mean_center=True, ksparse=None):
        """
        Parameters:
        ----------
        dim:            (int) dimensionality of data
        nsource:        (int) number of sources
        scale:          (float) scale of laplacians,
                        only relevant relative to noise scale
        nonneg:         if True, draw from exponential instead of laplace
        nstims:         (int) number of data to generate
        rng:            (int) random seed, default 912017
        batch_size:     (int) default batch size for rand_stim
        white:          if True, PCA-whiten after generation
        noise:          (float) scale of gaussian noise
        mean_center:    if True, center data after generation
        ksparse:        (int or None) number of sources generating each datum
        """
        self.stimshape = [dim]
        self.stimsize = dim
        self.nstims = nstims
        self.batch_size = batch_size
        self.datasize = dim

        if rng is None:
            rng = np.random.RandomState(912017)
        self.rng = rng

        # generate data
        if nsource is None:
            nsource = dim
        self.sources = rng.normal(size=(nsource, dim))
        self.sources /= np.linalg.norm(self.sources, axis=1, keepdims=True)

        self.scale = scale
        coefficients = rng.laplace(scale=self.scale, size=(self.nstims, nsource))
        self.nonneg = nonneg
        if self.nonneg:
            coefficients = np.abs(coefficients)
        if ksparse is not None:
            ktrues = np.array([True]*ksparse + [False]*(nsource - ksparse))
            mask = [np.random.permutation(ktrues) for _ in range(nstims)]
            mask = np.vstack(mask)
            coefficients[~mask] = 0
        self.data = coefficients.dot(self.sources)

        if noise > 0:
            self.data += rng.normal(scale=noise, size=self.data.shape)

        if mean_center:
            self.data -= self.data.mean(0, keepdims=True)

        self.white = white
        if white:
            self.whiten()

        stddev = self.data.std(0, keepdims=True)
        self.data /= stddev
        self.sparsity_parameter = noise**2 / stddev.mean()

    def test_fit(self, model):
        """Given a model for the sources, calculate a distance metric from
        true sources. The metric is the median of the normalized dot products
        between each true source and the closest model source."""
        fit = model/np.linalg.norm(model, axis=1, keepdims=True)
        if self.white:
            sources = self.sources.dot(self.zca_matrix)
            sources /= np.linalg.norm(sources, axis=1, keepdims=True)
        else:
            sources = self.sources
        allthedots = sources.dot(fit.T)
        # how close is the closest model source to each true source?
        bestfits = np.max(np.abs(allthedots), axis=1)
        return np.median(bestfits)

    def whiten(self, blocks=20000, eps=0.0001):
        """Assumes self.data already created, mean 0."""
        cov = np.zeros([self.datasize, self.datasize])
        nblocks = int(np.ceil(self.nstims / blocks))
        for ind in range(nblocks):
            X = self.data[blocks*ind:blocks*(ind+1)]
            cov += X.T.dot(X)
        eigvals, eigvecs = np.linalg.eigh(cov)
        if np.any(np.isnan(eigvals)):
            print('Warning: some nan eigenvalues found, replacing with small numbers.')
            eigvals[np.isnan(eigvals)] = 0.9 * eps**2
        if np.any(eigvals < 0):
            print('Warning: some negative eigenvalues of covariance matrix found. Replacing with small numbers.')
            eigvals[eigvals < 0] = 0.9 * eps**2

        idx = np.argsort(eigvals)
        svals = np.sqrt(eigvals[idx][::-1])
        eigvecs = eigvecs[idx][::-1]

        # do ZCA whitening
        wm = np.diag(1./np.maximum(svals, eps))
        self.zca_matrix = eigvecs.T.dot(wm).dot(eigvecs)
        self.data = self.data.dot(self.zca_matrix)
