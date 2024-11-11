
import numpy as np
import matplotlib.pyplot as plt

def generate_data(params):
	x = np.linspace(-0.35, 0.55, params.num_samples)
	x_noise = np.random.normal(0., params.x_noise_std, size=x.shape)
	y_noise = np.random.normal(0., params.y_noise_std, size=x.shape)
	y = x + 0.3 * np.sin(2 * np.pi * (x + x_noise)) + 0.3 * np.sin(4 * np.pi * (x + x_noise)) + y_noise
	x, y = x.reshape(-1, 1), y.reshape(-1, 1)
	mu = np.array([[-0.3,-0.3],[-0.18, -0.8],[0,0], [0.15, 0.8], [0.35,0.3], [0.55, 0.6]])
	if True:
		plt.scatter(x, y, alpha=0.5)
		plt.scatter(mu[:,0], mu[:,1], color='red', marker='o', s=100)
		plt.show()
	return x, y
