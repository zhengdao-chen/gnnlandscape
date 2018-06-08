import numpy as np
# import scipy as sp
from numpy.linalg import inv
from scipy.linalg import eigh, cholesky, eigvalsh, norm
import matplotlib.pyplot as plt

def get_adjacency_matrix(n, p, q):
	A = np.zeros([n * 2, n * 2])
	A[:n, :n] = np.random.binomial(1, p, (n, n))
	A[n:, n:] = np.random.binomial(1, p, (n, n))
	# A[:n, n:] = np.random.binomial(1, q, (n, n))
	A[n:, :n] = np.random.binomial(1, q, (n, n))
	for i in range(n * 2):
		for j in range(i+1, n * 2):
			A[i, j] = A[j, i]
	A = A * (np.ones(n * 2) - np.eye(n * 2))	# set the diagonal to zeros 
	return A

def get_Lambda(w, d, c):
    n_doubled = w.shape[0]
    Lambda = np.zeros([n_doubled, d])
    for k in range(d):
        Lambda[:, k] = np.multiply(c, pow(w, k))
    return Lambda

def get_Friedler(n):
	frdl = np.zeros([n * 2, 1])
	frdl[:n] = 1
	frdl[n:] = -1
	return frdl

def get_X_and_Y(n, d, w, v):
	c = np.random.randn(n * 2)
	c = c / np.linalg.norm(c, 2)

	Lambda = get_Lambda(w, d, c)

	X = np.transpose(Lambda).dot(Lambda)

	frdl = get_Friedler(n)
	b = np.transpose(v).dot(frdl) / np.sqrt(2 * n)

	# Y = ((np.transpose(Lambda).dot(b)).dot(np.transpose(b))).dot(Lambda)
	intermediate = np.transpose(Lambda).dot(b)
	Y = intermediate.dot(np.transpose(intermediate))

	# X = X / ((p + q) * n)
	# Y = Y / ((p + q) * n)

	return Lambda, intermediate, X, Y


# n = 900
# n_lst = [4, 16, 64, 256, 1024, 4096]
n_lst = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]	# different (half-)sizes of the graph

# lists for storing the quantities we are interested in, as sequences indexed by n
lst_1 = []
lst_2 = []
lst_3 = []
lst_3p = []
lst_4 = []

for n in n_lst:
	d = 5
	p = 0.8
	q = 0.2
	n_samples = 10000

	M = get_adjacency_matrix(n, p, q)

	w, v = eigh(M)

	X_lst = []
	Y_lst = []
	A_lst = []
	B_lst = []
	deltaB_lst = []

	avgX = np.zeros(d)
	avgY = np.zeros(d)
	avgA = np.zeros(d)
	avgB = np.zeros(d)

	for i in range(n_samples):
		Lambda, intermediate, X, Y = get_X_and_Y(n, d, w, v)
		X_lst.append(X)
		Y_lst.append(Y)
		avgX = avgX + X
		avgY = avgY + Y
	avgX = avgX / n_samples
	avgY = avgY / n_samples

	T = cholesky(avgX, lower=True)
	invT = inv(T)

	for i in range(n_samples):
		X = X_lst[i]
		Y = Y_lst[i]
		A = invT.dot(Y).dot(np.transpose(invT))
		B = invT.dot(X).dot(np.transpose(invT))
		avgA = avgA + A
		avgB = avgB + B
		deltaB = B - np.eye(d)
		A_lst.append(A)
		B_lst.append(B)
		deltaB_lst.append(deltaB)
	avgA = avgA / n_samples
	avgB = avgB / n_samples

	w_avgA = eigvalsh(avgA)
	quant_1 = w_avgA[-1] - w_avgA[-2]
	print ('quantity 1', quant_1)

	quant_2 = 0
	for i in range(n_samples):
		# w_A = eigvalsh(A_lst[i])
		l1A = eigvalsh(A_lst[i], eigvals=(d-1, d-1))[0]
		quant_2 = quant_2 + pow(abs(l1A), 6)
	quant_2 = quant_2 / n_samples
	print ('quantity 2', quant_2)

	quant_3 = 0
	for i in range(n_samples):
		norm_deltaB = norm(deltaB_lst[i], ord=2)
		quant_3 = quant_3 + pow(norm_deltaB, 6)
		# print (norm_deltaB)
	quant_3 = quant_3 / n_samples
	print ('quantity 3', quant_3)

	# changing the power on which norm_deltaB is raised from 6 to 2
	quant_3p = 0
	for i in range(n_samples):
		norm_deltaB = norm(deltaB_lst[i], ord=2)
		quant_3p = quant_3p + pow(norm_deltaB, 2)
		# print (norm_deltaB)
	quant_3p = quant_3p / n_samples
	print ('quantity 3 prime', quant_3p)

	quant_4 = 0
	for i in range(n_samples):
		lMB = eigvalsh(B_lst[i], eigvals=(0, 0))[0]
		# print (lMB)
		quant_4 = quant_4 + pow(abs(1 / lMB), 6)
	quant_4 = quant_4 / n_samples
	print ('quantity 4', quant_4)

	lst_1.append(quant_1)
	lst_2.append(quant_2)
	lst_3.append(quant_3)
	lst_3p.append(quant_3p)
	lst_4.append(quant_4)

plt.loglog(n_lst, lst_1)
plt.xlabel('n')
plt.ylabel('quantity 1: lambda_1(EA) - lambda_2(EA)')
plt.title('Quantity 1 as a function of n (p = 0.8, q = 0.2, M = 5)')
plt.savefig('./plots/quant_1_666_3.pdf')
plt.show()

plt.loglog(n_lst, lst_2)
plt.xlabel('n')
plt.ylabel('quantity 2: E[|lambda_1(A)|^6]')
plt.title('Quantity 2 as a function of n (p = 0.8, q = 0.2, M = 5)')
plt.savefig('./plots/quant_2_666_3.pdf')
plt.show()

plt.loglog(n_lst, lst_3)
plt.xlabel('n')
plt.ylabel('quantity 3: E[||DeltaB||^6]')
plt.title('Quantity 3 as a function of n (p = 0.8, q = 0.2, M = 5)')
plt.savefig('./plots/quant_3_666_3.pdf')
plt.show()

plt.loglog(n_lst, lst_3p)
plt.xlabel('n')
plt.ylabel('quantity 3p: E[||DeltaB||^2]')
plt.title('Quantity 3p as a function of n (p = 0.8, q = 0.2, M = 5)')
plt.savefig('./plots/quant_3p_666_3.pdf')
plt.show()

plt.loglog(n_lst, lst_4)
plt.xlabel('n')
plt.ylabel('quantity 4: E[1/|lambda_M(B)|^6]')
plt.title('Quantity 4 as a function of n (p = 0.8, q = 0.2, M = 5)')
plt.savefig('./plots/quant_4_666_3.pdf')
plt.show()



