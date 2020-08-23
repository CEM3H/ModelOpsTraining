# %%
import pandas as pd
import logging

from sklearn.datasets import load_breast_cancer
from pathlib import Path


def main():
    logger = logging.getLogger(__name__)
    logger.info('Loading breast cancer data from sklearn.datasets')
    df = get_data()

    logger.info('Saving data to ../data/raw/data.csv')
    save_data(df)


def get_data():
    """ Load breast cancer data from sklearn datasets
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')

    df = pd.concat([X, y], axis=1, sort=False)

    return df


def save_data(df):
    p = project_dir / 'data' / 'raw' / 'data.csv'
    df.to_csv(p, sep=';', decimal='.', index=None)


# %%
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    main()
