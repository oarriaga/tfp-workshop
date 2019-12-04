import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from datasets import WorldRuggedness


def TrainablePrior(kernel_size, bias_size=0, dtype=None):
    """Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
    This is learning the prior probabilities of each weight.
    Therefore it predicts an N dimensional Gaussian.
    This Gaussian is independent of the batch (maybe the features)?
    It learns an N bias vector that is used as mean for the Gaussians.
    """
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1)),
    ])


def PosteriorMeanField(kernel_size, bias_size=0, dtype=None):
    """Specify the surrogate posterior over `keras.layers.Dense`
    `kernel` and `bias`.
    This is learning the posterior probabilities of each weight
    """
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda x: tfd.Independent(
            tfd.Normal(x[..., :n], 1e-5 + tf.nn.softplus(c + x[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])


def LinearRegressor(KL_weight):
    """Instantiates a linear regression model using Keras API
    # Arguments
        batch_shape: List of Ints e.g. [batch_size, num_features]
        build_distribution: Python function.
    """
    model = tf.keras.Sequential([
        tfp.layers.DenseVariational(
            2, PosteriorMeanField, TrainablePrior, KL_weight),
        tfp.layers.DistributionLambda(lambda x: tfd.Normal(
            loc=x[..., :1],
            scale=1e-3 + tf.math.softplus(0.01 * x[..., 1:])))])
    return model


def negative_log_likelihood(y_true, predicted_distributions):
    """Calculates the negative log likelihood of the predicted distribution
    ``predicted_distribution`` and the true label value ``y_true``
    # Arguments
        y_true: Numpy array of shape [num_samples, 1].
        predicted_distribution: TensorFlow probability distribution
    """
    log_likelihood = predicted_distributions.log_prob(y_true)
    return - log_likelihood


# loading dataset
data_manager = WorldRuggedness()
dataset = data_manager.load_data()
(GDP_non_african, ruggedness_non_african), (GDP_african, ruggedness_african) = dataset

# plotting raw data
figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
sns.scatterplot(GDP_non_african, ruggedness_non_african, ax=axes[0])
sns.scatterplot(GDP_african, ruggedness_african, ax=axes[1])
xlabel, ylabel, = 'Terrain Ruggedness Index', 'log GDP per capita (2000)'
axes[0].set(xlabel=xlabel, ylabel=ylabel, title='Non African Nations')
axes[1].set(xlabel=xlabel, ylabel=ylabel, title='African Nations')

# iterating over non African and African splits
for axis, data in enumerate(dataset):
    # instantiating model, loss and optimizer
    model = LinearRegressor(1 / len(data[0]))
    model.compile(tf.optimizers.Adam(0.01), negative_log_likelihood)
    model.fit(*data, epochs=2000, verbose=2)

    # calculating mean and standard deviations of predicted distribution
    x_test = np.linspace(0, 6, 1000)[..., np.newaxis]
    y_hats = [model(x_test) for _ in range(100)]
    average_mean = np.zeros_like(x_test[..., 0])
    for experiment_arg, y_hat in enumerate(y_hats):
        mean = np.squeeze(y_hat.mean())
        standard_deviation = np.squeeze(y_hat.stddev())
        two_standard_deviations = 2 * standard_deviation
        if experiment_arg < 15:
            axes[axis].plot(
                x_test, mean, 'r', label='ensemble means', linewidth=1.0)
            axes[axis].plot(
                x_test, mean + two_standard_deviations, 'g',
                linewidth=0.5, label='ensemble means + 2 ensemble stdev')
            axes[axis].plot(
                x_test, mean - two_standard_deviations, 'g',
                linewidth=0.5, label='ensemble means - 2 ensemble stdev')
        average_mean = average_mean + mean
    axes[axis].plot(x_test, average_mean / len(y_hats), 'r',
                    label='overall mean', linewidth=2.0)
plt.show()
