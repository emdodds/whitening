# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 12:01:18 2015

@author: Eric Dodds

Abstract dictionary learner.
Includes gradient descent on MSE energy function as a default learning method.
"""
import numpy as np
import pickle
# the try/except block avoids an issue with the cluster
try:
    import matplotlib.pyplot as plt
    from scipy import ndimage
    from scipy.stats import skew
except ImportError:
    print('Plotting and modulation plot unavailable.')
import StimSet


class DictLearner(object):
    """Abstract base class for dictionary learner objects. Provides some
    default functions for loading data, plotting network properties,
    and learning."""
    def __init__(self, data, learnrate, nunits,
                 paramfile=None, theta=0, moving_avg_rate=0.001,
                 stimshape=None, datatype="image", batch_size=100, pca=None,
                 store_every=1):

        self.nunits = nunits
        self.batch_size = batch_size
        self.learnrate = learnrate
        self.paramfile = paramfile
        self.theta = theta
        self.moving_avg_rate = moving_avg_rate
        self.initialize_stats()
        self.store_every = store_every

        self._load_stims(data, datatype, stimshape, pca)

        self.Q = self.rand_dict()

    def initialize_stats(self):
        nunits = self.nunits
        self.corrmatrix_ave = np.zeros((nunits, nunits))
        self.L0hist = np.array([])
        self.L1hist = np.array([])
        self.L2hist = np.array([])
        self.L0acts = np.zeros(nunits)
        self.L1acts = np.zeros(nunits)
        self.L2acts = np.zeros(nunits)
        self.errorhist = np.array([])
        self.meanacts = np.zeros_like(self.L0acts)

    def _load_stims(self, data, datatype, stimshape, pca):
        if isinstance(data, StimSet.StimSet):
            self.stims = data
        elif datatype == "image" and pca is not None:
            stimshape = stimshape or (16, 16)
            self.stims = StimSet.PCvecSet(data, stimshape, pca,
                                          self.batch_size)
        elif datatype == "image":
            stimshape = stimshape or (16, 16)
            self.stims = StimSet.ImageSet(data, batch_size=self.batch_size,
                                          buffer=20, stimshape=stimshape)
        elif datatype == "spectro" and pca is not None:
            if stimshape is None:
                raise Exception("When using PC representations, \
                    you need to provide the shape of the original stimuli.")
            self.stims = StimSet.SpectroPCSet(data, stimshape, pca,
                                              self.batch_size)
        elif datatype == "waveform" and pca is not None:
            self.stims = StimSet.WaveformPCSet(data, stimshape, pca,
                                               self.batch_size)
        else:
            raise ValueError("Specified data type not currently supported.")

    def infer(self, data, infplot):
        raise NotImplementedError

    def test_inference(self, niter=None):
        """Show perfomance of infer() on a random batch."""
        temp = self.niter
        self.niter = niter or self.niter
        X = self.stims.rand_stim()
        s = self.infer(X, infplot=True)[0]
        self.niter = temp
        print("Final SNR: " + str(self.snr(X, s)))
        return s

    def generate_model(self, acts):
        """Reconstruct inputs using linear generative model."""
        return np.dot(self.Q.T, acts)

    def compute_errors(self, acts, X):
        """Given a batch of data and activities, compute the squared error between
        the generative model and the original data.
        Returns vector of mean squared errors."""
        diffs = X - self.generate_model(acts)
        return np.mean(diffs**2, axis=0)/np.mean(X**2, axis=0)

    def smoothed_error(self, window_size=1000, start=0, end=-1):
        """Plots a moving average of the error history
        with the given averaging window."""
        window = np.ones(int(window_size))/float(window_size)
        smoothed = np.convolve(self.errorhist[start:end], window, 'valid')
        plt.plot(smoothed)

    def progress_plot(self, window_size=1000, norm=1, start=0, end=-1):
        """Plots a moving average of the error and activity history
        with the given averaging window."""
        window = np.ones(int(window_size))/float(window_size)
        smoothederror = np.convolve(self.errorhist[start:end], window, 'valid')
        if norm == 2:
            acthist = self.L2hist
        elif norm == 0:
            acthist = self.L0hist
        else:
            acthist = self.L1hist
        smoothedactivity = np.convolve(acthist[start:end], window, 'valid')
        plt.plot(smoothederror, 'b', smoothedactivity, 'g')

    def snr(self, data, acts):
        """Returns signal-noise ratio for the given data and coefficients."""
        sig = np.var(data, axis=0)
        noise = np.var(data - self.Q.T.dot(acts), axis=0)
        return np.mean(sig/noise)

    def learn(self, data, coeffs, normalize=True):
        """Adjust dictionary elements according to gradient descent on the
        mean-squared error energy function, optionally with an extra term to
        increase orthogonality between basis functions. This term is
        multiplied by the parameter theta.
        Returns the mean-squared error."""
        R = data.T - np.dot(coeffs.T, self.Q)
        self.Q = self.Q + self.learnrate*np.dot(coeffs, R)
        if self.theta != 0:
            # Notice this is calculated using the Q after the mse learning rule
            thetaterm = (self.Q - np.dot(self.Q, np.dot(self.Q.T, self.Q)))
            self.Q = self.Q + self.theta*thetaterm
        if normalize:
            # force dictionary elements to be normalized
            normmatrix = np.diag(1./np.sqrt(np.sum(self.Q*self.Q, 1)))
            self.Q = normmatrix.dot(self.Q)
        return np.mean(R**2)

    def run(self, ntrials=1000, batch_size=None,
            rate_decay=None, normalize=True):
        batch_size = batch_size or self.stims.batch_size
        for trial in range(ntrials):
            X = self.stims.rand_stim(batch_size=batch_size)
            acts, _, _ = self.infer(X)
            thiserror = self.learn(X, acts, normalize)

            if trial % self.store_every == 0:
                if trial % 50 == 0 or self.store_every > 50:
                    print(trial)
                self.store_statistics(acts, thiserror, batch_size)

            if (trial % 1000 == 0 or trial+1 == ntrials) and trial != 0:
                try:
                    print("Saving progress to " + self.paramfile)
                    self.save()
                except (ValueError, TypeError) as er:
                    print('Failed to save parameters. ', er)
            if rate_decay is not None:
                self.adjust_rates(rate_decay)

    def store_statistics(self, acts, thiserror, batch_size=None,
                         center_corr=True):
        batch_size = batch_size or self.batch_size
        self.L2acts = ((1-self.moving_avg_rate)*self.L2acts +
                       self.moving_avg_rate*(acts**2).mean(1))
        self.L1acts = ((1-self.moving_avg_rate)*self.L1acts +
                       self.moving_avg_rate*np.abs(acts).mean(1))
        L0means = np.mean(acts != 0, axis=1)
        self.L0acts = ((1-self.moving_avg_rate)*self.L0acts +
                       self.moving_avg_rate*L0means)
        means = acts.mean(1)
        self.meanacts = ((1-self.moving_avg_rate)*self.meanacts +
                         self.moving_avg_rate*means)
        self.errorhist = np.append(self.errorhist, thiserror)
        self.L0hist = np.append(self.L0hist, np.mean(acts != 0))
        self.L1hist = np.append(self.L1hist, np.mean(np.abs(acts)))
        self.L2hist = np.append(self.L2hist, np.mean(acts**2))
        return self.compute_corrmatrix(acts, thiserror,
                                       means, center_corr, batch_size)

    def compute_corrmatrix(self, acts, thiserror, means,
                           center_corr=True, batch_size=None):
        batch_size = batch_size or self.batch_size
        if center_corr:
            actdevs = acts-means[:, np.newaxis]
            corrmatrix = (actdevs).dot(actdevs.T)/batch_size
        else:
            corrmatrix = acts.dot(acts.T)/self.batch_size
        self.corrmatrix_ave = ((1-self.moving_avg_rate)*self.corrmatrix_ave +
                               self.moving_avg_rate*corrmatrix)
        return corrmatrix

    def skewflip(self):
        """Set each dictionary element to minus itself if the skewness
        of its linear projections on a large batch of data is negative."""
        dots = np.dot(self.Q, self.stims.rand_stim(batch_size=10000))
        mask = skew(dots, axis=1) < 0
        self.Q[mask] = - self.Q[mask]

    def show_dict(self, stimset=None, cmap='RdBu_r', subset=None,
                  layout='sqrt', savestr=None):
        """Plot an array of tiled dictionary elements.
        The 0th element is in the top right."""
        stimset = stimset or self.stims
        if subset is not None:
            indices = np.random.choice(self.Q.shape[0], subset)
            Qs = self.Q[np.sort(indices)]
        else:
            Qs = self.Q
        array = stimset.stimarray(Qs[::-1], layout=layout)
        plt.figure()
        arrayplot = plt.imshow(array, interpolation='nearest', cmap=cmap,
                               aspect='auto', origin='lower')
        plt.axis('off')
        plt.colorbar()
        if savestr is not None:
            plt.savefig(savestr, bbox_inches='tight')
        return arrayplot

    def tiled_dict(self, cmap='RdBu_r', layout='sqrt',
                   aspect='auto', savestr=None):
        """Nicer dictionary visualization.
        Creates a matplotlib axis for each element, so very slow."""
        self.stims.tiledplot(self.Q, cmap=cmap, layout=layout,
                             aspect=aspect, savestr=savestr)

    def show_element(self, index, cmap='jet', labels=None, savestr=None):
        elem = self.stims.stim_for_display(self.Q[index])
        plt.figure()
        plt.imshow(elem.T, interpolation='nearest', cmap=cmap,
                   aspect='auto', origin='lower')
        if labels is None:
            plt.axis('off')
        else:
            plt.colorbar()
        if savestr is not None:
            plt.savefig(savestr, bbox_inches='tight')

    def rand_dict(self):
        Q = np.random.randn(self.nunits, self.stims.datasize)
        return (np.diag(1/np.sqrt(np.sum(Q**2, 1)))).dot(Q)

    def adjust_rates(self, factor):
        """Multiply the learning rate by the given factor."""
        self.learnrate = factor*self.learnrate
        self.theta = factor*self.theta

    def modulation_plot(self, usepeaks=False, **kwargs):
        modcentroids = np.zeros((self.Q.shape[0], 2))
        for ii in range(self.Q.shape[0]):
            modspec = self.stims.modspec(self.Q[ii])
            if usepeaks:
                modcentroids[ii, 0] = np.argmax(np.mean(modspec, axis=1))
                modcentroids[ii, 1] = np.argmax(np.mean(modspec, axis=0))
            else:
                modcentroids[ii] = ndimage.measurements.center_of_mass(modspec)
        plt.scatter(modcentroids[:, 0], modcentroids[:, 1])
        plt.title('Center of mass of modulation power spectrum \
            of each dictionary element')
        try:
            plt.xlabel(kwargs.xlabel)
        except:
            pass
        try:
            plt.ylabel(kwargs.ylabel)
        except:
            pass

    def sort_dict(self, batch_size=None,
                  plot=False, allstims=True, savestr=None):
        """Sorts the RFs in order by their usage on a batch. Default batch size
        is 10 times the stored batch size. Usage means 1 for each stimulus for
        which the element was used and 0 for the other stimuli, averaged over
        stimuli."""
        if allstims:
            testX = self.stims.data.T
        else:
            batch_size = batch_size or 10*self.batch_size
            testX = self.stims.rand_stim(batch_size)
        means = np.mean(self.infer(testX)[0] != 0, axis=1)
        sorter = np.argsort(means)
        self.sort(means, sorter, plot, savestr)
        return means[sorter]

    def fast_sort(self, L1=False, plot=False, savestr=None):
        """Sorts RFs in order by moving average usage."""
        if L1:
            usages = self.L1acts
        else:
            usages = self.L0acts
        sorter = np.argsort(usages)
        self.sort(usages, sorter, plot, savestr)
        return usages[sorter]

    def sort(self, usages, sorter, plot=False, savestr=None):
        self.Q = self.Q[sorter]
        self.L0acts = self.L0acts[sorter]
        self.L1acts = self.L1acts[sorter]
        self.L2acts = self.L2acts[sorter]
        self.meanacts = self.meanacts[sorter]
        self.corrmatrix_ave = self.corrmatrix_ave[sorter]
        self.corrmatrix_ave = self.corrmatrix_ave.T[sorter].T
        if plot:
            plt.figure()
            plt.plot(usages[sorter])
            plt.title('L0 Usage')
            plt.xlabel('Dictionary index')
            plt.ylabel('Fraction of stimuli')
            if savestr is not None:
                plt.savefig(savestr, format='png', bbox_inches='tight')

    def load(self, filename=None):
        if filename is None:
            filename = self.paramfile
        self.paramfile = filename
        with open(filename, 'rb') as f:
            self.Q, params, histories = pickle.load(f)
        self.set_histories(histories)
        self.set_params(params)

    def set_params(self, params):
        for key, val in params.items():
            try:
                getattr(self, key)
            except AttributeError:
                print('Unexpected parameter passed: ' + key)
            setattr(self, key, val)

    def get_param_list(self):
        raise NotImplementedError

    def save(self, filename=None):
        filename = filename or self.paramfile
        if filename is None:
            raise ValueError("You need to input a filename.")
        self.paramfile = filename
        params = self.get_param_list()
        histories = self.get_histories()
        with open(filename, 'wb') as f:
            pickle.dump([self.Q, params, histories], f)

    def get_histories(self):
        return {'errorhist': self.errorhist,
                'L0hist': self.L0hist,
                'L1hist': self.L1hist,
                'L2hist': self.L2hist,
                'corrmatrix_ave': self.corrmatrix_ave,
                'L1': self.L1hist,
                'L0acts': self.L0acts,
                'L1acts': self.L1acts,
                'L2acts': self.L2acts,
                'meanacts': self.meanacts}

    def set_histories(self, histories):
        if not isinstance(histories, dict):
            self._old_set_histories(histories)
            return
        self.errorhist = histories['errorhist']
        self.L0hist = histories['L0hist']
        self.L1hist = histories['L1hist']
        self.L2hist = histories['L2hist']
        self.corrmatrix_ave = histories['corrmatrix_ave']
        self.L1hist = histories['L1']
        self.L0acts = histories['L0acts']
        self.L1acts = histories['L1acts']
        self.L2acts = histories['L2acts']
        self.meanacts = histories['meanacts']

    def _old_get_histories(self):
        return (self.errorhist, self.meanacts, self.L0acts, self.L0hist,
                self.L1acts, self.L1hist, self.L2hist, self.L2acts,
                self.corrmatrix_ave)

    def _old_set_histories(self, histories):
        (self.errorhist, self.meanacts, self.L0acts, self.L0hist,
         self.L1acts, self.L1hist, self.L2hist, self.L2acts,
         self.corrmatrix_ave) = histories
