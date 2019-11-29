import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


URL = 'https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv'
data = pd.read_csv(URL, encoding='ISO-8859-1')
original_labels = ['cont_africa', 'rugged', 'rgdppc_2000']
labels = ['africa', 'ruggedness', 'GDP_per_capita']

data = data[original_labels]
data = data.rename(columns=dict(zip(original_labels, labels)))
data['GDP_per_capita'] = np.log(data['GDP_per_capita'])
data = data[np.isfinite(data.GDP_per_capita)]

african_nations = data[data['africa'] == 1]
non_african_nations = data[data['africa'] == 0]

# plotting
figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
sns.scatterplot(non_african_nations['ruggedness'],
                non_african_nations['GDP_per_capita'],
                ax=axes[0])

axes[0].set(xlabel='Terrain Ruggedness Index',
            ylabel='log GDP per capita (2000)',
            title='Non African Nations')

sns.scatterplot(african_nations['ruggedness'],
                african_nations['GDP_per_capita'],
                ax=axes[1])

axes[1].set(xlabel='Terrain Ruggedness Index',
            ylabel='log GDP per capita (2000)',
            title='African Nations')


# NLL = lambda y, predicted_distribution: -predicted_distribution.log_prob(y)
def negative_log_likelihood(y_true, predicted_distributions):
    log_likelihood = predicted_distributions.log_prob(y_true)
    return - log_likelihood


for data_arg, data in enumerate([non_african_nations, african_nations]):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2),
        tfp.layers.DistributionLambda(
            lambda x: tfd.Normal(
                x[..., :1],
                scale=1e-3 + tf.math.softplus(0.05 * x[..., 1:])))])

    x = data.ruggedness.to_numpy()
    y = data.GDP_per_capita.to_numpy()
    model.compile(tf.optimizers.Adam(0.01), negative_log_likelihood)
    model.fit(x, y, epochs=1000, verbose=True)
    x_test = np.linspace(0, 6, 1000)
    predicted_distribution = model(x_test[..., np.newaxis])

    mean = predicted_distribution.mean()
    standard_deviation = predicted_distribution.stddev()
    positive_standard_deviation = mean + 2. * standard_deviation
    negative_standard_deviation = mean - 2. * standard_deviation
    axes[data_arg].plot(x_test, mean, 'r--', label='mean')
    axes[data_arg].plot(x_test, positive_standard_deviation, 'g-', label='mean + 2 stddev')
    axes[data_arg].plot(x_test, negative_standard_deviation, 'g-', label='mean  -  2 stddev')
    axes[data_arg].legend()

plt.show()
