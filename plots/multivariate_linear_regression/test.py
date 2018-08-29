import sys
import matplotlib.pyplot as plt
from random import randint

data_plain_text = open("data.txt", "r")												 	 			 #load data to memory
data_matrix = []

for i in data_plain_text:
	data_matrix += [[1.0] + list(map(lambda a: float(a), i.split(" ")))] #insert dummy input for weights[0]

weights = [0 for i in range(6)]
#weights = [0.0006930449991439476, 0.018092900052509107, 0.00514462606296179, 0.00010200223803102637, 0.033865004926948813, 7.819526052013582e-06]
#weights = [1.000386469038675, 0.014372872686644802, 1.0030966344631689, 1.000058142636903, 1.0170036581920823, 1.0000047464723236]

a = 0.00000001 * (1/(len(data_matrix)*0.8))													 			 #and one for the w_0*x_0 intercept
lam = 0.001

#print(data_matrix[1])
#print(weights)

def h(weights, x):																		 			 #regression hypothesis
	return sum([weights[i]*x[i] for i in range(len(weights))])

w = weights
x = data_matrix[:int(len(data_matrix)*0.8)]												 			 #train for the first 500 training-samples

def train(m):
	global weights
	global w
	global x
	global lam

	LENGTH = len(data_matrix) - len(x)
	errors = []

	def plot(xs, ys):
		plt.plot([0] + xs, [0] + ys, label="With regularization")
		plt.xlabel('Iterations')
		plt.ylabel('Mean squared loss')	
		#plt.savefig('regularizederrorplot.pgf')
		#plt.show()

	def L(q, w):																		 #q-regularization function
		return sum([abs(w[i])**q for i in range(len(w))])

	for n in range(m):
		sys.stdout.write("\r" + ("%.2f" % (n/m * 100)))
		if n%1 == 0:
			error = 0	

			for i in range(len(x), len(data_matrix)):												     
				error += abs((data_matrix[i][6] - h(weights, data_matrix[i])))**2	 #L_2 summed loss function

			errors += [error/LENGTH]
			
			#print("mean error: ", error/LENGTH)	
			"""
			print("weights: ", weights)
			print("________________________________________________________________________________________________")
			"""

		for i in range(len(weights)):													 #update weights using gradient descent
			#w[i] = w[i] + a * sum([x[j][i] * (x[j][6] - h(w, x[j][:6])) for j in range(len(x))]) 
			w[i] = w[i] + a * sum([x[j][i] * (x[j][6] - h(w, x[j][:6])) for j in range(len(x))]) + lam
		
		weights = w
			
	print("mean squared loss: ", errors[-1])	
	print(weights)
	plot(list(range(m)), errors)

train(1000)
##############

weights = [0 for i in range(6)]
#weights = [0.0006930449991439476, 0.018092900052509107, 0.00514462606296179, 0.00010200223803102637, 0.033865004926948813, 7.819526052013582e-06]
#weights = [1.000386469038675, 0.014372872686644802, 1.0030966344631689, 1.000058142636903, 1.0170036581920823, 1.0000047464723236]

a = 0.00000001 * (1/(len(data_matrix)*0.8))													 			 #and one for the w_0*x_0 intercept
lam = 0.001

#print(data_matrix[1])
#print(weights)

def h(weights, x):																		 			 #regression hypothesis
	return sum([weights[i]*x[i] for i in range(len(weights))])

w = weights
x = data_matrix[:int(len(data_matrix)*0.8)]												 			 #train for the first 500 training-samples

def train(m):
	global weights
	global w
	global x
	global lam

	LENGTH = len(data_matrix) - len(x)
	errors = []

	def plot(xs, ys):
		plt.plot([0] + xs, [0] + ys, label="Without regularization")
		#plt.xlabel('Iterations')
		#plt.ylabel('Mean squared loss')	
		#plt.savefig('regularizederrorplot.pgf')
		#plt.show()

	def L(q, w):																		 #q-regularization function
		return sum([abs(w[i])**q for i in range(len(w))])

	for n in range(m):
		sys.stdout.write("\r" + ("%.2f" % (n/m * 100)))
		if n%1 == 0:
			error = 0	

			for i in range(len(x), len(data_matrix)):												     
				error += abs((data_matrix[i][6] - h(weights, data_matrix[i])))**2	 #L_2 summed loss function

			errors += [error/LENGTH]
			
			#print("mean error: ", error/LENGTH)	
			"""
			print("weights: ", weights)
			print("________________________________________________________________________________________________")
			"""

		for i in range(len(weights)):													 #update weights using gradient descent
			w[i] = w[i] + a * sum([x[j][i] * (x[j][6] - h(w, x[j][:6])) for j in range(len(x))]) 
			#w[i] = w[i] + a * sum([x[j][i] * (x[j][6] - h(w, x[j][:6])) for j in range(len(x))]) + lam
		
		weights = w
			
	print("mean squared loss: ", errors[-1])	
	print(weights)
	plot(list(range(m)), errors)

train(1000)

#if sys.argv[1] == 'train':
#	train(int(sys.argv[2]))
#
#elif sys.argv[1] == 'test':
#	for i in x:
#		print("guessed: ", h(weights, i[:6]), " , expected: ", i[6])
#		
plt.legend(loc="best")
plt.show()