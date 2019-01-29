import numpy as np

import chainer
import chainer.distributions as D
import chainer.functions as F
import chainer.links as L
from chainer import reporter

import net as N_


################################################################################

class VAELoss(chainer.Chain, N_.AEBase):

    def __init__(self, predictor, beta=1.0, k=1):
        super().__init__()
        self.beta = beta
        self.k = k
        n_latent = predictor.n_latent

        with self.init_scope():
            # self.link = predictor
            self.predictor = predictor
            self.prior = Prior(n_latent)

        self.n_latent = n_latent
        self.adjust()

    def __call__(self, x, x_=None, **kwargs):
        if x_ is None:
            x_ = x

        q_z = self.encode(x, **kwargs)
        z = self.sample(q_z)
        p_x = self.decode(z, **kwargs)
        p_z = self.prior()
        x_ext = F.repeat(x_, self.k, axis=0)

        # def to_a(v):
        #     return chainer.cuda.to_cpu(v.array)

        reconstr = self.batch_mean(p_x.log_prob(x_ext))
        kl_penalty = self.batch_mean(chainer.kl_divergence(q_z, p_z))
        loss = self.beta * kl_penalty - reconstr
        reporter.report({'loss': loss}, self)
        reporter.report({'reconstr': reconstr}, self)
        reporter.report({'kl_penalty': kl_penalty}, self)
        return loss

    def encode(self, x, **kwargs):
        q_z = self.predictor.encode(x, **kwargs)
        return q_z

    def decode(self, x, **kwargs):
        y = self.predictor.decode(x, **kwargs)
        if kwargs.get('inference'):
            return F.sigmoid(y)
        p_x = D.Bernoulli(logit=y)
        return p_x

    def sample(self, q_z):
        z = q_z.sample(self.k)
        return z.reshape((-1, *z.shape[2:]))

    def batch_mean(self, v):
        return F.mean(F.sum(v, axis=tuple(range(1, v.ndim))))

    def predict(self, x, **kwargs):
        xa = self.predictor.xp.asarray(x)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            return F.sigmoid(self.predictor(xa, inference=True, **kwargs))

    @property
    def link(self):
        return self.predictor


################################################################################

class VAEChain(chainer.Chain, N_.AEBase):
    ''' 単層エンコーダ+デコーダ(VAE全結合ネットワーク)
    '''

    def __init__(self, in_size, out_size, activation=F.sigmoid):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.in_shape = None
        self.init = False
        self.maybe_init(in_size)

        if type(activation) is tuple:
            self.activation_e = None
            self.activation_d = activation[1]
        else:
            self.activation_e = None
            self.activation_d = activation

    def __call__(self, x, **kwargs):
        h = self.encode(x, **kwargs)
        y = self.decode(h, **kwargs)
        return y

    def encode(self, x, **kwargs):
        self.in_shape = x.shape[1:]
        self.maybe_init(self.in_shape)

        x_ = x.reshape(-1, self.in_size)
        mu = self.mu(x_)

        if kwargs.get('show_shape'):
            print(f'layer(E{self.name}): in: {x.shape} out: {mu.shape}')

        if kwargs.get('inference'):
            return mu # == D.Normal(loc=mu, log_scale=ln_sigma).mean
        else:
            ln_sigma = self.ln_sigma(x_)  # log(sigma)
            return D.Normal(loc=mu, log_scale=ln_sigma)

    def decode(self, x, **kwargs):
        if isinstance(x, D.Normal):
            x = x.mean
        y = self.dec(x)
        if self.activation_d:
            y = self.activation_d(y)
        y = y.reshape(-1, *self.in_shape)

        if kwargs.get('show_shape'):
            print(f'layer(D{self.name}): in: {x.shape} out: {y.shape}')
        return y

    def maybe_init(self, in_size_):
        if self.init:
            return
        elif in_size_:
            if type(in_size_) is tuple:
                in_size = np.prod(in_size_)
            else:
                in_size = in_size_

            with self.init_scope():
                self.mu = L.Linear(in_size, self.out_size)
                self.ln_sigma = L.Linear(in_size, self.out_size)
                self.dec = L.Linear(self.out_size, in_size)

            self.in_size = in_size
            self.init = True
            self.adjust()


################################################################################

class AvgELBOLoss(chainer.Chain):
    """Loss function of VAE.

    The loss value is equal to ELBO (Evidence Lower Bound)
    multiplied by -1.

    Args:
        encoder (chainer.Chain): A neural network which outputs variational
            posterior distribution q(z|x) of a latent variable z given
            an observed variable x.
        decoder (chainer.Chain): A neural network which outputs conditional
            distribution p(x|z) of the observed variable x given
            the latent variable z.
        prior (chainer.Chain): A prior distribution over the latent variable z.
        beta (float): Usually this is 1.0. Can be changed to control the
            second term of ELBO bound, which works as regularization.
        k (int): Number of Monte Carlo samples used in encoded vector.
    """

    def __init__(self, encoder, decoder, prior, beta=1.0, k=1):
        super(AvgELBOLoss, self).__init__()
        self.beta = beta
        self.k = k

        with self.init_scope():
            self.encoder = encoder
            self.decoder = decoder
            self.prior = prior

    def __call__(self, x):
        q_z = self.encoder(x)
        z = q_z.sample(self.k)
        p_x = self.decoder(z)
        p_z = self.prior()

        reconstr = F.mean(F.sum(p_x.log_prob(
            F.broadcast_to(x[None, :], (self.k,) + x.shape)), axis=-1))
        kl_penalty = F.mean(F.sum(chainer.kl_divergence(q_z, p_z), axis=-1))
        loss = - (reconstr - self.beta * kl_penalty)
        reporter.report({'loss': loss}, self)
        reporter.report({'reconstr': reconstr}, self)
        reporter.report({'kl_penalty': kl_penalty}, self)
        return loss


class Encoder(chainer.Chain):

    def __init__(self, n_in, n_latent, n_h):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.linear = L.Linear(n_in, n_h)
            self.mu = L.Linear(n_h, n_latent)
            self.ln_sigma = L.Linear(n_h, n_latent)

    def forward(self, x):
        h = F.tanh(self.linear(x))
        mu = self.mu(h)
        ln_sigma = self.ln_sigma(h)  # log(sigma)
        return D.Normal(loc=mu, log_scale=ln_sigma)


class Decoder(chainer.Chain):

    def __init__(self, n_in, n_latent, n_h, binary_check=False):
        super(Decoder, self).__init__()
        self.binary_check = binary_check
        with self.init_scope():
            self.linear = L.Linear(n_latent, n_h)
            self.output = L.Linear(n_h, n_in)

    def forward(self, z, inference=False):
        n_batch_axes = 1 if inference else 2
        h = F.tanh(self.linear(z, n_batch_axes=n_batch_axes))
        h = self.output(h, n_batch_axes=n_batch_axes)
        return D.Bernoulli(logit=h, binary_check=self.binary_check)


class Prior(chainer.Link):

    def __init__(self, n_latent):
        super(Prior, self).__init__()

        self.loc = np.zeros(n_latent, np.float32)
        self.scale = np.ones(n_latent, np.float32)
        self.register_persistent('loc')
        self.register_persistent('scale')

    def forward(self):
        return D.Normal(self.loc, scale=self.scale)


def make_encoder(n_in, n_latent, n_h):
    return Encoder(n_in, n_latent, n_h)


def make_decoder(n_in, n_latent, n_h, binary_check=False):
    return Decoder(n_in, n_latent, n_h, binary_check=binary_check)


def make_prior(n_latent):
    return Prior(n_latent)
