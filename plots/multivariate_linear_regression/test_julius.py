import random, numpy, os

DATA = []

with open("data.txt", "r") as h:
	lines = h.read().split("\n")
	for l in lines:
		DATA += [[1.0] + list(map(lambda a: float(a), l.split(" ")))]				 #dummy input for weights[0]

#weights = [1 for i in range(6)]
WEIGHTS = [0.05699923528160729, 
		   0.0007752241469733542, 
		   0.34679162085400184, 
		   0.015159271209071484, 
		   2.039697851913147, 
		   0.00048204826809488186]		

ALPHA = 0.00001 * (1/len(DATA) * 0.8)

def h(weights, x):
	return sum([weights[i]*x[i] for i in range(len(weights))])

X = DATA[:int(len(DATA) * 0.8)]
X_TRAIN = DATA[int(len(DATA) * 0.2):]

def train(weights, alpha, m=100):

	a = alpha
	w = weights

	def check_err(weights):
		error = 0
		
		for xt in X_TRAIN: 
			error += abs(xt[6] - h(w, xt[:6]))
		
		error /= len(X_TRAIN)

		return error


	for n in range(m):		

		for i in range(len(w)):
			w[i] += a * sum([X[j][i] * (X[j][6] - h(w, X[j][:6])) for j in range(len(X))]) 
		
		error = check_err(w)

		a *= error / 30


		print ("ERROR:", error, a)

train(WEIGHTS, ALPHA, 10**10)