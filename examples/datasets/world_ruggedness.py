import pandas as pd
import numpy as np

URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"


class WorldRuggedness(object):
    """Data used in [1] for estimating correlation between GDP vs. Ruggedness.

    # Methods
        load_data

    # References
        [1]  Ruggedness:The blessing of bad geography in Africa
    """
    def __init__(self, URL=URL):
        self.URL = URL

    def load_data(self):
        """ Loads data from URL and preprocess by removing NaNs and
            setting GDP in natural log scale.

        # Returns
            Two lists containing GDP per capita and ruggedness coefficient
            for non African nations and African nations respectively.
            e.g. [(gdp_non_african, r_non_african), (gdp_african, r_african)]
        """
        data_frame = pd.read_csv(self.URL, encoding='ISO-8859-1')
        data_frame = data_frame[['cont_africa', 'rugged', 'rgdppc_2000']]
        data_frame.rgdppc_2000 = np.log(data_frame.rgdppc_2000)
        data_frame = data_frame[np.isfinite(data_frame.rgdppc_2000)]
        datasets = []
        for flag in [0, 1]:
            split = data_frame[data_frame.cont_africa == flag]
            x, y = split.rugged.to_numpy(), split.rgdppc_2000.to_numpy()
            datasets.append([x, y])
        return datasets
