import unittest

import numpy as np

import lab7

n_input = 2
n_hidden = 50
n_output = 2

class TestNN(unittest.TestCase):
	def test_initialize_weights(self):
		nn = lab7.NN(n_input, n_hidden, n_output)

		self.assertEqual(nn.W1.shape, (2,50))
		self.assertEqual(nn.b1.shape, (50,))
		self.assertEqual(nn.W2.shape, (50,2))
		self.assertEqual(nn.b2.shape, (2,))

	def test_softmax(self):
		nn = lab7.NN(n_input, n_hidden, n_output)

		correct_probs = np.array([[2.689e-1, 7.311e-1], [2.143e-24, 1.0000], [8.808e-1, 1.192e-1]])
		probs = np.array([nn.softmax(x) for x in [[1,2], [-10,44.5], [0, -2]]])

		self.assertTrue(np.allclose(correct_probs, probs, atol=1e-4))

	def test_feed_forward(self):
		nn = lab7.NN(n_input, n_hidden, n_output)

		np.random.seed(0)
		nn.W1 = np.random.randn(n_input, n_hidden)
		nn.b1 = np.random.randn(n_hidden)
		nn.W2 = np.random.randn(n_hidden, n_output)
		nn.b2 = np.random.randn(n_output)

		correct_probs = np.array([[9.999e-1, 3.574e-7], [2.929e-5, 1.000], [9.945e-1, 5.480e-3]])
		probs = np.array([nn.feed_forward(x) for x in [[1,2], [-3,5.5], [0,0.01]]])

		self.assertEqual(probs.shape, (3,2))
		self.assertTrue(np.all(0 <= probs) and np.all(probs <= 1))
		self.assertTrue(np.allclose([1, 1, 1], np.sum(probs, axis=1), atol=1e-4))
		self.assertTrue(np.allclose(correct_probs, probs, atol=1e-4))

	def test_predict(self):
		nn = lab7.NN(n_input, n_hidden, n_output)

		np.random.seed(0)
		nn.W1 = np.random.randn(n_input, n_hidden)
		nn.b1 = np.random.randn(n_hidden)
		nn.W2 = np.random.randn(n_hidden, n_output)
		nn.b2 = np.random.randn(n_output)

		correct_predictions = np.array([0, 1, 0])
		predictions = np.array([nn.predict(x) for x in [[1,2], [-3,5.5], [0,0.01]]])

		self.assertEqual(predictions.shape, (3,))
		self.assertTrue(np.all(np.logical_or(predictions == 0, predictions == 1)))
		self.assertTrue(np.array_equal(correct_predictions, predictions))

	def test_compute_accuracy(self):
		nn = lab7.NN(n_input, n_hidden, n_output)

		np.random.seed(0)
		nn.W1 = np.random.randn(n_input, n_hidden)
		nn.b1 = np.random.randn(n_hidden)
		nn.W2 = np.random.randn(n_hidden, n_output)
		nn.b2 = np.random.randn(n_output)

		correct_accuracy = 2/3
		accuracy = nn.compute_accuracy(np.array([[1,2], [-3,5.5], [0,0.01]]), np.array([0, 1, 1]))

		self.assertTrue(0 <= accuracy <= 1)
		self.assertTrue(abs(correct_accuracy - accuracy) <= 1e-4)

if __name__ == '__main__':
	unittest.main()
