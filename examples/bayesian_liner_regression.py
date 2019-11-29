import tensorflow as tf
from tensorflow_probability import distributions as tfd


class BayesianLinearRegression(tf.keras.Model):
    def __init__(self, output_dim, name=None):
        super(BayesianLinearRegression, self).__init__(name=name)
        self.w_loc = tf.Variable(tf.random.normal([output_dim, 1]), name='w_loc')
        self.w_std = tf.Variable(tf.random.normal([output_dim, 1]), name='w_std')
        self.b_loc = tf.Variable(tf.random.normal([1]), name='b_loc')
        self.b_std = tf.Variable(tf.random.normal([1]), name='b_std')
        self.s_alpha = tf.Variable(tf.exp(tf.random.normal([1])), name='s_alpha')
        self.s_beta = tf.Variable(tf.exp(tf.random.normal([1])), name='s_beta')

    @property
    def weight(self):
        """Variational posterior for the weight"""
        return tfd.Normal(self.w_loc, tf.exp(self.w_std))

    @property
    def bias(self):
        """Variational posterior for the bias"""
        return tfd.Normal(self.b_loc, tf.exp(self.b_std))

    @property
    def std(self):
        """Variational posterior for the noise standard deviation"""
        return tfd.InverseGamma(tf.exp(self.s_alpha), tf.exp(self.s_beta))

    def call(self, x, sampling=True):
        """Predict p(y|x)"""
        sample = self._sample(x, sampling)
        loc = x @ sample(self.weight) + sample(self.bias)
        std = tf.sqrt(sample(self.std))
        return tfd.Normal(loc, std)

    def _sample(self, x, sampling=True):
        return x.sample if sampling else x.mean()

    @property
    def losses(self):
        """Sum of KL divergences between posteriors and priors"""
        prior = tfd.Normal(0, 1)
        return (tf.reduce_sum(tfd.kl_divergence(self.weight, prior)) +
                tf.reduce_sum(tfd.kl_divergence(self.bias, prior)))
