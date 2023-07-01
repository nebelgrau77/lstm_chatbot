import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

def sentences_stats(df, cols):
    '''returns some statictics about the sentences - intended to be ran after the sentences are tokenized'''
    
    for col in cols:
        print(f'Sentences in column {col}:\n\t \
        mean: {df[col].apply(lambda x: len(x)).mean():.2f}\n\t \
        median: {df[col].apply(lambda x: len(x)).median():.2f}\n\t \
        minimum: {df[col].apply(lambda x: len(x)).min()}\n\t \
        maximum: {df[col].apply(lambda x: len(x)).max()})')

def histograms(df, cols, name = ''):

    '''Display a histogram of the lengths of sentences with bins for each length, returns the bins and counts'''
    
    fig, ax = plt.subplots()

    xaxis_len = 0 # for the x-axis ticks

    data = {}
    
    for column, color in zip(cols, ['teal', 'purple']):
        
        values = df[column].apply(lambda x: len(x))

        print(column, values.min())
        
        counts, bins = np.histogram(values, bins = np.arange(values.max()))

        ax.hist(bins[:-1], bins, weights = counts, color = color, alpha = 0.5, label = column, density = True)

        if values.max() > xaxis_len:
            xaxis_len = values.max()

        data[column] = [counts, bins]
    
    ax.set_title(f'Sentences length {name}')
    ax.set_xlabel('length [tokens]')
    ax.set_ylabel('frequency')
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.legend()
    ax.set_xticks([_ for _ in range(0, xaxis_len+1, 2)])
    
    plt.show()
    
    return data