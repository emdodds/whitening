import numpy as np
from SAILnet import SAILmods


class bSAILnet(SAILmods.VarTimeSAILnet):
    """SAILnet mod with the inhibitory connection learning rule replaced by the
    rule of Brendel et al., which acts to balance excitation and inhibition for
    each neuron."""

    def infer_with_updateW(self, X, infplot=False, savestr=None):
        """
        Simulate LIF neurons to code X.
        """
        nstim = X.shape[-1]

        # projections of stimuli onto feedforward weights
        B = np.dot(self.Q, X)

        u = np.zeros((self.nunits, nstim))  # internal unit variables
        y = np.zeros((self.nunits, nstim))  # external unit variables
        acts = np.zeros((self.nunits, nstim))  # counts of total firings

        dW = np.zeros_like(self.W)

        if infplot:
            errors = np.zeros(self.niter)
            yhist = np.zeros((self.niter))

        for t in range(self.niter):

            # DE for internal variables
            u = (1.-self.infrate)*u + self.infrate*(B - self.W.dot(y))

            # external variables spike when internal variables cross thresholds
            y = np.array([u[:, ind] >= self.theta for ind in range(nstim)]).T

            acts += y

            # accumulate W update
            dW += u.dot(y.T) - self.infrate*self.W*(y.sum(axis=1)[None, :])

            # reset the internal variables of the spiking units
            # notice this happens after W update, so spiking units may still have incoming inhibition increased
            u[y] = 0

            if infplot:
                recon_t = self.gain*acts/((t+1)*self.infrate)
                errors[t] = np.mean(self.compute_errors(recon_t, X))
                yhist[t] = np.mean(y)

        if infplot:
            self.plotter.inference_plots(errors, yhist, savestr=savestr)

        self.W += self.alpha*dW/nstim
        self.W[self.W < 0] = 0
        for ii in range(self.nunits):
            self.W[ii,ii] = 0

        return self.gain*acts/self.inftime

    def learn_no_lateral(self, X, acts):
        # update feedforward weights with Oja's rule
        sumsquareacts = np.sum(acts*acts, 1)  # square, then sum over images
        dQ = acts.dot(X.T) - np.diag(sumsquareacts).dot(self.Q)
        self.Q = self.Q + self.beta*dQ/self.batch_size

        # update thresholds with Foldiak's rule: keep firing rates near target
        dtheta = self.gamma*(np.sum(acts, 1)/self.batch_size - self.p)
        self.theta = self.theta + dtheta

        if self.gain_rate > 0:
            self.gain = self.gain*(self.compute_gain(X, acts)/self.gain)**self.gain_rate

    def run(self, ntrials=25000, rate_decay=1):
        """
        Run bSAILnet for ntrials: for each trial, create a random set of image
        patches, present each to the network, and update the network weights
        after each set of batch_size presentations.
        The learning rates are multiplied by rate_decay after each trial.
        """
        for t in range(ntrials):
            X = self.stims.rand_stim()

            #acts, u, y = self.infer_with_history(X)
            acts = self.infer_with_updateW(X)
            errors = np.mean(self.compute_errors(acts, X))
            if t % self.store_every == 0:
                corrmatrix = self.store_statistics(acts, errors)
                self.objhistory = np.append(self.objhistory,
                                            self.compute_objective(acts, X))
            else:
                corrmatrix = self.compute_corrmatrix(acts, errors, acts.mean(1))

            #self.learn_with_history(X, acts, u, y)
            self.learn_no_lateral(X, acts)

            if t % 50 == 0:
                print("Trial number: " + str(t))
                if t % 5000 == 0:
                    # save progress
                    print("Saving progress...")
                    self.save()
                    print("Done. Continuing to run...")

            self.adjust_rates(rate_decay)

        self.save()

    def test_inference(self):
        X = self.stims.rand_stim()
        temp = self.alpha
        self.alpha = 0
        s = self.infer_with_updateW(X, infplot=True)
        self.alpha = temp
        print("Final SNR: " + str(self.snr(X, s)))
        return s


class cov_bSAILnet(bSAILnet):
    """bSAILnet with the feedforward weight learning rule Brendel et al. suggest
    for unwhitened data. The rule essentially accounts for the covariance of the data."""

    def learn_no_lateral(self, X, acts):
        dQ = acts.dot(X.T) - self.Q.dot(X).dot(X.T)
        self.Q = self.Q + self.beta*dQ/self.batch_size

        # update thresholds with Foldiak's rule: keep firing rates near target
        dtheta = self.gamma*(np.sum(acts, 1)/self.batch_size - self.p)
        self.theta = self.theta + dtheta

        if self.gain_rate > 0:
            self.gain = self.gain*(self.compute_gain(X, acts)/self.gain)**self.gain_rate


class mpbSAILnet(bSAILnet):
    """An attempt to approximate cov_bSAILnet using only the membrane potential
    in the feedforward learning rule, as opposed to the aggregated ff input."""

    def __init__(self, *args, norm = False, **kwargs):
        bSAILnet.__init__(self, *args, **kwargs)
        self.finite_time_factor = 1/(1 + (np.exp(-self.inftime) - 1)/self.inftime)
        self.norm = norm

    def learn_no_lateral(self, X, acts):
        # update thresholds with Foldiak's rule: keep firing rates near target
        dtheta = self.gamma*(np.sum(acts, 1)/self.batch_size - self.p)
        self.theta = self.theta + dtheta

        if self.gain_rate > 0:
            self.gain = self.gain*(self.compute_gain(X, acts)/self.gain)**self.gain_rate

    def infer_with_updateW(self, X, infplot=False, savestr=None):
        """
        Simulate LIF neurons to code X.
        """
        nstim = X.shape[-1]

        # projections of stimuli onto feedforward weights
        B = np.dot(self.Q, X)

        u = np.zeros((self.nunits, nstim))  # internal unit variables
        y = np.zeros((self.nunits, nstim))  # external unit variables
        acts = np.zeros((self.nunits, nstim))  # counts of total firings

        dW = np.zeros_like(self.W)
        dQ = np.zeros_like(self.Q)

        if infplot:
            errors = np.zeros(self.niter)
            yhist = np.zeros((self.niter))

        for t in range(self.niter):

            # DE for internal variables
            u = (1.-self.infrate)*u + self.infrate*(B - self.W.dot(y))

            # external variables spike when internal variables cross thresholds
            y = np.array([u[:, ind] >= self.theta for ind in range(nstim)]).T

            acts += y

            # accumulate W and Q updates
            dW += u.dot(y.T) - self.infrate*self.W*(y.sum(axis=1)[None, :])
            #dQ += y.dot(X.T) - (u/self.theta[:,None]).dot(X.T)/self.niter
            dQ += y.dot(X.T) - self.finite_time_factor*u.dot(X.T)/self.niter

            # reset the internal variables of the spiking units
            # notice this happens after W update, so spiking units may still have incoming inhibition increased
            u[y] = 0

            if infplot:
                recon_t = self.gain*acts/((t+1)*self.infrate)
                errors[t] = np.mean(self.compute_errors(recon_t, X))
                yhist[t] = np.mean(y)

        if infplot:
            self.plotter.inference_plots(errors, yhist, savestr=savestr)

        self.W += self.alpha*dW/nstim
        self.W[self.W < 0] = 0
        for ii in range(self.nunits):
            self.W[ii, ii] = 0

        self.Q += self.beta*dQ/nstim
        if self.norm:
            self.Q = (np.diag(1/np.sqrt(np.sum(self.Q**2, 1)))).dot(self.Q)

        return self.gain*acts/self.inftime

    def infer_with_histories(self, X, infplot=False, savestr=None):
        nstim = X.shape[-1]

        # projections of stimuli onto feedforward weights
        B = np.dot(self.Q, X)

        u = np.zeros((self.nunits, nstim))  # internal unit variables
        y = np.zeros((self.nunits, nstim))  # external unit variables
        acts = np.zeros((self.nunits, nstim))  # counts of total firings

        uhistory = np.zeros([self.nunits, nstim, self.niter])

        if infplot:
            errors = np.zeros(self.niter)
            yhist = np.zeros((self.niter))

        for t in range(self.niter):

            # DE for internal variables
            u = (1.-self.infrate)*u + self.infrate*(B - self.W.dot(y))
            uhistory[:, :, t] = u

            # external variables spike when internal variables cross thresholds
            y = np.array([u[:, ind] >= self.theta for ind in range(nstim)]).T

            acts += y

            # reset the internal variables of the spiking units
            # notice this happens after W update, so spiking units may still have incoming inhibition increased
            u[y] = 0

            if infplot:
                recon_t = self.gain*acts/((t+1)*self.infrate)
                errors[t] = np.mean(self.compute_errors(recon_t, X))
                yhist[t] = np.mean(y)

        if infplot:
            self.plotter.inference_plots(errors, yhist, savestr=savestr)

        return self.gain*acts/self.inftime, uhistory


class BalanceNet(bSAILnet):
    # status: mysteriously blows up after some apparently successful training

    def initialize(self, *args):
        bSAILnet.initialize(self, *args)
        self.inhibscale = 2 - self.p/self.theta  # Brendel S.36
        self.excitscale = (2*self.theta - self.p) / np.sum(self.Q * self.Q, axis=1)

    def infer_with_updateW(self, X, infplot=False, savestr=None):
        """
        Simulate LIF neurons to code X.
        """
        nstim = X.shape[-1]

        # projections of stimuli onto feedforward weights
        B = np.dot(self.Q, X)

        u = np.zeros((self.nunits, nstim))  # internal unit variables
        y = np.zeros((self.nunits, nstim))  # external unit variables
        acts = np.zeros((self.nunits, nstim))  # counts of total firings

        dW = np.zeros_like(self.W)

        if infplot:
            errors = np.zeros(self.niter)
            yhist = np.zeros((self.niter))

        for t in range(self.niter):

            # DE for internal variables
            u = (1.-self.infrate)*u + self.infrate*(B - self.W.dot(y))

            # external variables spike when internal variables cross thresholds
            y = np.array([u[:, ind] >= self.theta for ind in range(nstim)]).T

            acts += y

            # accumulate W update
            dW += self.inhibscale*u.dot(y.T) - self.infrate*self.W*(y.sum(axis=1)[None, :])

            if infplot:
                recon_t = self.gain*acts/((t+1)*self.infrate)
                errors[t] = np.mean(self.compute_errors(recon_t, X))
                yhist[t] = np.mean(y)

            # reset the internal variables of the spiking units
            u[y] = 0

        if infplot:
            self.plotter.inference_plots(errors, yhist, savestr=savestr)

        self.W += self.alpha*dW/nstim

        return self.gain*acts/self.inftime

    def learn_no_lateral(self, X, acts):
        # note theta is constant

        dQ = self.excitscale*acts.dot(X.T) - self.Q.dot(acts*X).dot(X.T)
        self.Q = self.Q + self.beta*dQ/self.batch_size
        #self.Q = np.diag(1/np.sqrt(np.sum(self.Q**2, 1))).dot(self.Q)

        self.inhibscale = 2 - self.p/self.theta  # Brendel S.36
        denom = np.sum(self.Q*acts.dot(X.T), axis=1)
        self.excitscale = (2*self.theta - self.p) / denom
