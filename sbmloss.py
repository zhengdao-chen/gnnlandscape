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

def plot_hist(arr):
	n_rows = arr.shape[0]
	vertical_dim = int(np.ceil(n_rows / 2))
	# fig, axs = plt.subplots(vertical_dim, 2, sharey=True, tight_layout=True)
	for i in range(n_rows):
		ax = plt.subplot(vertical_dim, 2, i+1)
		# axs[i % vertical_dim][int(np.floor(i / vertical_dim))].hist(arr[i], bins=100)
		ax.hist(arr[i], bins=100)
		# ax.set_xscale('log')
		# axs.xscale('log')
	# axs[1].hist(y, bins=n_bins)

	plt.show()
	return 0


# n = 900
# n_lst = [4, 16, 64, 256, 1024, 4096]
n_lst = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]	# different (half-)sizes of the graph
# n_lst = [4, 16, 64, 256, 1024]

# lists for storing the quantities we are interested in, as sequences indexed by n
lst_1 = []
lst_2 = []
lst_3 = []
lst_3p = []
lst_4 = []

n_samples = 10000

# arr_1 = np.zeros([len(n_lst), n_samples])
arr_2 = np.zeros([len(n_lst), n_samples])
arr_3 = np.zeros([len(n_lst), n_samples])
arr_4 = np.zeros([len(n_lst), n_samples])

for j in range(len(n_lst)):
	n = n_lst[j]
	d = 5
	p = 0.8
	q = 0.2

	M = get_adjacency_matrix(n, p, q)
	M = M / ((p + q) * n)

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
		arr_2[j, i] = abs(l1A)
		quant_2 = quant_2 + pow(abs(l1A), 6)
	quant_2 = quant_2 / n_samples
	print ('quantity 2', quant_2)

	quant_3 = 0
	for i in range(n_samples):
		norm_deltaB = norm(deltaB_lst[i], ord=2)
		arr_3[j, i] = norm_deltaB
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
		arr_4[j, i] = abs(1 / lMB)
		# print (lMB)
		quant_4 = quant_4 + pow(abs(1 / lMB), 6)
	quant_4 = quant_4 / n_samples
	print ('quantity 4', quant_4)

	lst_1.append(quant_1)
	lst_2.append(quant_2)
	lst_3.append(quant_3)
	lst_3p.append(quant_3p)
	lst_4.append(quant_4)

mean_q2_2nd = np.mean(pow(arr_2, 2), 1)
mean_q2_3rd = np.mean(pow(arr_2, 3), 1)
mean_q2_6th = np.mean(pow(arr_2, 6), 1)
mean_q3_2nd = np.mean(pow(arr_3, 2), 1)
mean_q3_3rd = np.mean(pow(arr_2, 3), 1)
mean_q3_6th = np.mean(pow(arr_3, 6), 1)
mean_q4_2nd = np.mean(pow(arr_4, 2), 1)
mean_q4_3rd = np.mean(pow(arr_2, 3), 1)
mean_q4_6th = np.mean(pow(arr_4, 6), 1)

std_q2_2nd = np.std(pow(arr_2, 2), 1)
std_q2_3rd = np.std(pow(arr_2, 3), 1)
std_q2_6th = np.std(pow(arr_2, 6), 1)
std_q3_2nd = np.std(pow(arr_3, 2), 1)
std_q3_3rd = np.std(pow(arr_2, 3), 1)
std_q3_6th = np.std(pow(arr_3, 6), 1)
std_q4_2nd = np.std(pow(arr_4, 2), 1)
std_q4_3rd = np.std(pow(arr_2, 3), 1)
std_q4_6th = np.std(pow(arr_4, 6), 1)

# plt.loglog(n_lst, lst_1)
# plt.xlabel('n')
# plt.ylabel('quantity 1: lambda_1(EA) - lambda_2(EA)')
# plt.title('Quantity 1 as a function of n (p = 0.8, q = 0.2, M = 5)')
# plt.savefig('./plots/quant_1_666_3.pdf')
# plt.show()

# plt.loglog(n_lst, lst_2)
# plt.xlabel('n')
# plt.ylabel('quantity 2: E[|lambda_1(A)|^6]')
# plt.title('Quantity 2 as a function of n (p = 0.8, q = 0.2, M = 5)')
# plt.savefig('./plots/quant_2_666_3.pdf')
# plt.show()

# plt.loglog(n_lst, lst_3)
# plt.xlabel('n')
# plt.ylabel('quantity 3: E[||DeltaB||^6]')
# plt.title('Quantity 3 as a function of n (p = 0.8, q = 0.2, M = 5)')
# # plt.savefig('./plots/quant_3_666_3.pdf')
# plt.show()

# plt.loglog(n_lst, lst_3p)
# plt.xlabel('n')
# plt.ylabel('quantity 3p: E[||DeltaB||^2]')
# plt.title('Quantity 3p as a function of n (p = 0.8, q = 0.2, M = 5)')
# plt.savefig('./plots/quant_3p_666_3.pdf')
# plt.show()

# plt.loglog(n_lst, lst_4)
# plt.xlabel('n')
# plt.ylabel('quantity 4: E[1/|lambda_M(B)|^6]')
# plt.title('Quantity 4 as a function of n (p = 0.8, q = 0.2, M = 5)')
# plt.savefig('./plots/quant_4_666_3.pdf')
# plt.show()

e = np.std(pow(arr_3, 6), 1)
plt.errorbar(np.log(n_lst), np.log(lst_3), np.log(e))
plt.xlabel('n')
plt.ylabel('quantity 3: E[||DeltaB||^6]')
plt.title('Quantity 3 as a function of n (p = 0.8, q = 0.2, M = 5)')
# plt.savefig('./plots/quant_3_666_3.pdf')
plt.show()



