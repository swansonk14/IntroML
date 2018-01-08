import sys
sys.path.append('../../')

import numpy as np

import lab1

def check_reviews_data(reviews_data):
    assert len(reviews_data) == 2000
    assert sorted(reviews_data[0].keys()) == sorted(['helpfulN', 'helpfulY', 'productId', 'sentiment', 'summary', 'text', 'userId'])

def check_toy_data(toy_data, toy_labels):
    assert toy_data.shape == (200, 2)
    assert toy_labels.shape == (200,)
    assert np.array_equal(np.sum(toy_data, axis=0), np.array([ 185.6102,  202.9751]))
    assert set(toy_labels) == {-1.0, 1.0}

if __name__ == '__main__':
    # Load reviews data
    reviews_data = lab1.load_reviews_data('../../Data/reviews_train.csv')
    check_reviews_data(reviews_data)
    print('Reviews data loaded correctly')

    # Load toy data
    toy_data, toy_labels = lab1.load_toy_data('../../Data/toy_data.csv')
    check_toy_data(toy_data, toy_labels)
    print('Toy data loaded correctly')

    # Plot toy data
    lab1.plot_toy_data(toy_data, toy_labels)
