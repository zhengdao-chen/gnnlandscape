import numpy as np
# import scipy as sp
from numpy.linalg import inv
from scipy.linalg import eigh, cholesky, eigvalsh, norm
# import matplotlib.pyplot as plt

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

def get_adjacency_matrix_new(n, p, q):
	p_prime = 1 - np.sqrt(1 - p)
	q_prime = 1 - np.sqrt(1 - q)
	A = np.zeros([n * 2, n * 2])
	A[:n, :n] = np.random.binomial(1, p, (n, n))
	A[n:, n:] = np.random.binomial(1, p, (n, n))
	A[:n, n:] = np.random.binomial(1, q, (n, n))
	A[n:, :n] = np.random.binomial(1, q, (n, n))
	A = A * (np.ones(n * 2) - np.eye(n * 2))
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

def plot_hist_q2(arr, a=0, b=8, c=0, d=50000):
	n_rows = arr.shape[0]
	vertical_dim = int(np.ceil(n_rows / 2))
	# fig, axs = plt.subplots(vertical_dim, 2, sharey=True, tight_layout=True)
	for i in range(n_rows):
		ax = plt.subplot(vertical_dim, 2, i+1)
		# axs[i % vertical_dim][int(np.floor(i / vertical_dim))].hist(arr[i], bins=100)
		ax.hist(arr[i], bins=100)
		ax.axis([a, b, c, d])
		# ax.set_xscale('log')
		# axs.xscale('log')
		ax.set_title('n=' + str(n_lst[i] * 2))
	# axs[1].hist(y, bins=n_bins)
	plt.savefig('./plots/hist_l1A.pdf')
	plt.show()
	return 0

def plot_hist_q3(arr, a=0, b=8, c=0, d=50000):
	n_rows = arr.shape[0]
	vertical_dim = int(np.ceil(n_rows / 2))
	# fig, axs = plt.subplots(vertical_dim, 2, sharey=True, tight_layout=True)
	for i in range(n_rows):
		ax = plt.subplot(vertical_dim, 2, i+1)
		# axs[i % vertical_dim][int(np.floor(i / vertical_dim))].hist(arr[i], bins=100)
		ax.hist(arr[i], bins=100)
		ax.axis([a, b, c, d])
		# ax.set_xscale('log')
		# axs.xscale('log')
		ax.set_title('n=' + str(n_lst[i] * 2))
	# axs[1].hist(y, bins=n_bins)
	plt.savefig('./plots/hist_dB.pdf')
	plt.show()
	return 0

def plot_hist_q4(arr, a=0, b=8, c=0, d=50000):
	n_rows = arr.shape[0]
	vertical_dim = int(np.ceil(n_rows / 2))
	# fig, axs = plt.subplots(vertical_dim, 2, sharey=True, tight_layout=True)
	for i in range(n_rows):
		ax = plt.subplot(vertical_dim, 2, i+1)
		# axs[i % vertical_dim][int(np.floor(i / vertical_dim))].hist(arr[i], bins=100)
		ax.hist(arr[i], bins=np.arange(100) * b / 100)
		ax.axis([a, b, c, d])
		# ax.set_xscale('log')
		# axs.xscale('log')
		ax.set_title('n=' + str(n_lst[i] * 2))
	# axs[1].hist(y, bins=n_bins)
	plt.savefig('./plots/hist_lMBinv_broader.pdf')
	plt.show()
	return 0

def print_matrices(M_tensor, n_lst, number):
	np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
	for j in range(M_tensor.shape[0]):
		print ('n = ' + str(n_lst[j]))
		for i in range(number):
			print (np.matrix(M_tensor[j, i, :, :]))
			# "%0.2f" % np.matrix(M_tensor[j, i, :, :])

def compute_cond(M_tensor, n_lst):
	cond = np.zeros([M_tensor.shape[0], M_tensor.shape[1]])

print ('start')

# # n = 900
# n_lst = [4, 16, 64, 256, 1024, 4096]
# n_lst = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]	# different (half-)sizes of the graph
n_lst = [4, 8, 16, 32, 64, 128, 256, 512]
# n_lst = [1024]
# n_lst = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
# # n_lst = [4, 16, 64, 256, 1024]

print ('n_lst: ', n_lst)

# # lists for storing the quantities we are interested in, as sequences indexed by n
lst_1 = []
lst_2 = []
lst_3 = []
lst_3p = []
lst_4 = []

# n_samples = 100000
n_samples = 20000
# n_samples = 10000
# n_samples = 100

n_samples_saved = 100

d = 5
# p = 0.8
p = 5.0
# q = 0.2
q = 1.0

# arr_1 = np.zeros([len(n_lst), n_samples])
arr_2 = np.zeros([len(n_lst), n_samples])
arr_3 = np.zeros([len(n_lst), n_samples])
arr_4 = np.zeros([len(n_lst), n_samples])


X_tensor = np.zeros([len(n_lst), n_samples_saved, d, d])
Y_tensor = np.zeros([len(n_lst), n_samples_saved, d, d])
A_tensor = np.zeros([len(n_lst), n_samples_saved, d, d])
B_tensor = np.zeros([len(n_lst), n_samples_saved, d, d])

w_tensor = np.zeros([len(n_lst), n_samples_saved, n_lst[len(n_lst) - 1] * 2])

for j in range(len(n_lst)):
	n = n_lst[j]

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
		M = get_adjacency_matrix_new(n, p / n, q / n)
		# M = M / ((p + q) * n)

		w, v = eigh(M)

		if (i < n_samples_saved):
			w_tensor[j, i, :n_lst[j]*2] = w

		Lambda, intermediate, X, Y = get_X_and_Y(n, d, w, v)
		X_lst.append(X)
		Y_lst.append(Y)
		avgX = avgX + X
		avgY = avgY + Y
		if (i < n_samples_saved):
			X_tensor[j, i, :, :] = X
			Y_tensor[j, i, :, :] = Y

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

		if (i < n_samples_saved):
			A_tensor[j, i, :, :] = A
			B_tensor[j, i, :, :] = B

	avgA = avgA / n_samples
	avgB = avgB / n_samples

	print ('n: ', n_lst[j])

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
	# print ('quantity 3 prime', quant_3p)

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




# # plt.loglog(n_lst, lst_1)
# # plt.xlabel('n')
# # plt.ylabel('quantity 1: lambda_1(EA) - lambda_2(EA)')
# # plt.title('Quantity 1 as a function of n (p = 0.8, q = 0.2, M = 5)')
# # plt.savefig('./plots/quant_1_666_3.pdf')
# # plt.show()

# # plt.loglog(n_lst, lst_2)
# # plt.xlabel('n')
# # plt.ylabel('quantity 2: E[|lambda_1(A)|^6]')
# # plt.title('Quantity 2 as a function of n (p = 0.8, q = 0.2, M = 5)')
# # plt.savefig('./plots/quant_2_666_3.pdf')
# # plt.show()

# # plt.loglog(n_lst, lst_3)
# # plt.xlabel('n')
# # plt.ylabel('quantity 3: E[||DeltaB||^6]')
# # plt.title('Quantity 3 as a function of n (p = 0.8, q = 0.2, M = 5)')
# # # plt.savefig('./plots/quant_3_666_3.pdf')
# # plt.show()

# # plt.loglog(n_lst, lst_3p)
# # plt.xlabel('n')
# # plt.ylabel('quantity 3p: E[||DeltaB||^2]')
# # plt.title('Quantity 3p as a function of n (p = 0.8, q = 0.2, M = 5)')
# # plt.savefig('./plots/quant_3p_666_3.pdf')
# # plt.show()

# # plt.loglog(n_lst, lst_4)
# # plt.xlabel('n')
# # plt.ylabel('quantity 4: E[1/|lambda_M(B)|^6]')
# # plt.title('Quantity 4 as a function of n (p = 0.8, q = 0.2, M = 5)')
# # plt.savefig('./plots/quant_4_666_3.pdf')
# # plt.show()

# e = np.std(pow(arr_3, 6), 1)
# plt.errorbar(np.log(n_lst), np.log(lst_3), np.log(e))
# plt.xlabel('n')
# plt.ylabel('quantity 3: E[||DeltaB||^6]')
# plt.title('Quantity 3 as a function of n (p = 0.8, q = 0.2, M = 5)')
# # plt.savefig('./plots/quant_3_666_3.pdf')
# plt.show()

# np.save('./results/arr_2_10_new', arr_2)
# np.save('./results/arr_3_10_new', arr_3)
# np.save('./results/arr_4_10_new', arr_4)

# np.save('./results/arr_2_8_new_2', arr_2)
# np.save('./results/arr_3_8_new_2', arr_3)
# np.save('./results/arr_4_8_new_2', arr_4)

# arr_2_old = np.load('./results/arr_2_sparse_8.npy')
# arr_3_old = np.load('./results/arr_3_sparse_8.npy')
# arr_4_old = np.load('./results/arr_4_sparse_8.npy')

# arr_2 = np.concatenate((arr_2_old, arr_2), 1)
# arr_3 = np.concatenate((arr_3_old, arr_3), 1)
# arr_4 = np.concatenate((arr_4_old, arr_4), 1)

np.save('./results/arr_2_sparse_8_sr', arr_2)
np.save('./results/arr_3_sparse_8_sr', arr_3)
np.save('./results/arr_4_sparse_8_sr', arr_4)

np.save('./results/X_tensor_sparse_8_sr', X_tensor)
np.save('./results/Y_tensor_sparse_8_sr', Y_tensor)
np.save('./results/A_tensor_sparse_8_sr', A_tensor)
np.save('./results/B_tensor_sparse_8_sr', B_tensor)
np.save('./results/w_tensor_sparse_8_sr', w_tensor)

# arr_2 = np.load('./results/arr_2_12.npy')
# arr_3 = np.load('./results/arr_3_12.npy')
# arr_4 = np.load('./results/arr_4_12.npy')

mean_q2_2nd = np.mean(pow(arr_2, 2), 1)
mean_q2_3rd = np.mean(pow(arr_2, 3), 1)
mean_q2_6th = np.mean(pow(arr_2, 6), 1)
mean_q3_2nd = np.mean(pow(arr_3, 2), 1)
mean_q3_3rd = np.mean(pow(arr_3, 3), 1)
mean_q3_6th = np.mean(pow(arr_3, 6), 1)
mean_q4_2nd = np.mean(pow(arr_4, 2), 1)
mean_q4_3rd = np.mean(pow(arr_4, 3), 1)
mean_q4_6th = np.mean(pow(arr_4, 6), 1)

std_q2_2nd = np.std(pow(arr_2, 2), 1)
std_q2_3rd = np.std(pow(arr_2, 3), 1)
std_q2_6th = np.std(pow(arr_2, 6), 1)
std_q3_2nd = np.std(pow(arr_3, 2), 1)
std_q3_3rd = np.std(pow(arr_3, 3), 1)
std_q3_6th = np.std(pow(arr_3, 6), 1)
std_q4_2nd = np.std(pow(arr_4, 2), 1)
std_q4_3rd = np.std(pow(arr_4, 3), 1)
std_q4_6th = np.std(pow(arr_4, 6), 1)

# for quantity_index in [2, 3, 4]:
# 	for power in [2, 3, 6]:
# 		plt.errorbar(np.log(n_lst), )



# plt.errorbar(np.log(n_lst), mean_q2_2nd, std_q2_2nd)
# plt.xlabel('n (half size of the graph)')
# plt.ylabel('E[|lambda_1(A)|^2]')

x = np.zeros(len(n_lst) * n_samples)
for i in range(len(n_lst)):
	x[n_samples * i : n_samples * (i+1)] = i + 2

# y_2 = arr_2.reshape(1, len(n_lst) * n_samples)
# y_3 = arr_3.reshape(1, len(n_lst) * n_samples)
# y_4 = arr_4.reshape(1, len(n_lst) * n_samples)

# scatter plots

# plt.scatter(x, y_2, s=0.3)
# plt.xlabel('log of n (half size of the graph)')
# plt.title('Sampled values of |lambda_1(A_n)|')
# plt.savefig('./plots/scatter_l1A.png')
# plt.show()

# plt.scatter(x, y_3, s=0.3)
# plt.xlabel('log of n (half size of the graph)')
# plt.title('Sampled values of ||delta B_n||')
# plt.savefig('./plots/scatter_dB.png')
# plt.show()

# plt.scatter(x, y_4, s=0.3)
# plt.xlabel('log of n (half size of the graph)')
# plt.title('Sampled values of 1/|lambda_M(B_n)|')
# plt.savefig('./plots/scatter_lMBinv.png')
# plt.show()

# errorbar plots

# plt.errorbar(np.arange(12)+2, mean_q2_2nd, std_q2_2nd)
# plt.xlabel('log of n (half size of the graph)')
# plt.title('E[|lambda_1(A_n)|^2]')
# plt.savefig('./plots/errbar_l1A_2nd.pdf')
# plt.show()

# plt.errorbar(np.arange(12)+2, mean_q2_3rd, std_q2_3rd)
# plt.xlabel('log of n (half size of the graph)')
# plt.title('E[|lambda_1(A_n)|^3]')
# plt.savefig('./plots/errbar_l1A_3rd.pdf')
# plt.show()

# plt.errorbar(np.arange(12)+2, mean_q2_6th, std_q2_6th)
# plt.xlabel('log of n (half size of the graph)')
# plt.title('E[|lambda_1(A_n)|^6] (quantity 2)')
# plt.savefig('./plots/errbar_l1A_6th.pdf')
# plt.show()

# plt.errorbar(np.arange(12)+2, mean_q3_2nd, std_q3_2nd)
# plt.xlabel('log of n (half size of the graph)')
# plt.title('E[||delta B_n||^2]')
# plt.savefig('./plots/errbar_dB_2nd.pdf')
# plt.show()

# plt.errorbar(np.arange(12)+2, mean_q3_3rd, std_q3_3rd)
# plt.xlabel('log of n (half size of the graph)')
# plt.title('E[||delta B_n||^3]')
# plt.savefig('./plots/errbar_dB_3rd.pdf')
# plt.show()

# plt.errorbar(np.arange(12)+2, mean_q3_6th, std_q3_6th)
# plt.xlabel('log of n (half size of the graph)')
# plt.title('E[|||delta B_n||^6] (quantity 3)')
# plt.savefig('./plots/errbar_dB_6th.pdf')
# plt.show()

# plt.errorbar(np.arange(12)+2, mean_q4_2nd, std_q4_2nd)
# plt.xlabel('log of n (half size of the graph)')
# plt.title('E[1/|lambda_M(B_n)|^2]')
# plt.savefig('./plots/errbar_lMBinv_2nd.pdf')
# plt.show()

# plt.errorbar(np.arange(12)+2, mean_q4_3rd, std_q4_3rd)
# plt.xlabel('log of n (half size of the graph)')
# plt.title('E[1/|lambda_M(B_n)|^3]')
# plt.savefig('./plots/errbar_lMBinv_3rd.pdf')
# plt.show()

# plt.errorbar(np.arange(12)+2, mean_q4_6th, std_q4_6th)
# plt.xlabel('log of n (half size of the graph)')
# plt.title('E[1/|lambda_M(B_n)|^6] (quantity 4)')
# plt.savefig('./plots/errbar_lMBinv_6th.pdf')
# plt.show()

# plt.errorbar(np.arange(12)+2, np.log(mean_q4_2nd), np.log(std_q4_2nd))
# plt.xlabel('log of n (half size of the graph)')
# plt.title('log of E[1/|lambda_M(B_n)|^2]')
# plt.savefig('./plots/errbar_lMBinv_2nd_log.pdf')
# plt.show()

# plt.errorbar(np.arange(12)+2, np.log(mean_q4_3rd), np.log(std_q4_3rd))
# plt.xlabel('log of n (half size of the graph)')
# plt.title('log of E[1/|lambda_M(B_n)|^3]')
# plt.savefig('./plots/errbar_lMBinv_3rd_log.pdf')
# plt.show()

# plt.errorbar(np.arange(12)+2, np.log(mean_q4_6th), np.log(std_q4_6th))
# plt.xlabel('log of n (half size of the graph)')
# plt.title('log of E[1/|lambda_M(B_n)|^6] (quantity 4)')
# plt.savefig('./plots/errbar_lMBinv_6th_log.pdf')
# plt.show()

## no std plots

# plt.plot(np.arange(12)+2, mean_q2_2nd)
# plt.xlabel('log of n (half size of the graph)')
# plt.title('E[|lambda_1(A_n)|^2]')
# plt.savefig('./plots/plot_l1A_2nd.pdf')
# plt.show()

# plt.plot(np.arange(12)+2, mean_q2_3rd)
# plt.xlabel('log of n (half size of the graph)')
# plt.title('E[|lambda_1(A_n)|^3]')
# plt.savefig('./plots/plot_l1A_3rd.pdf')
# plt.show()

# plt.plot(np.arange(12)+2, mean_q2_6th)
# plt.xlabel('log of n (half size of the graph)')
# plt.title('E[|lambda_1(A_n)|^6] (quantity 2)')
# plt.savefig('./plots/plot_l1A_6th.pdf')
# plt.show()

# plt.plot(np.arange(12)+2, mean_q3_2nd)
# plt.xlabel('log of n (half size of the graph)')
# plt.title('E[||delta B_n||^2]')
# plt.savefig('./plots/plot_dB_2nd.pdf')
# plt.show()

# plt.plot(np.arange(12)+2, mean_q3_3rd)
# plt.xlabel('log of n (half size of the graph)')
# plt.title('E[||delta B_n||^3]')
# plt.savefig('./plots/plot_dB_3rd.pdf')
# plt.show()

# plt.plot(np.arange(12)+2, mean_q3_6th)
# plt.xlabel('log of n (half size of the graph)')
# plt.title('E[|||delta B_n||^6] (quantity 3)')
# plt.savefig('./plots/plot_dB_6th.pdf')
# plt.show()

# plt.plot(np.arange(12)+2, mean_q4_2nd)
# plt.xlabel('log of n (half size of the graph)')
# plt.title('E[1/|lambda_M(B_n)|^2]')
# plt.savefig('./plots/plot_lMBinv_2nd.pdf')
# plt.show()

# plt.plot(np.arange(12)+2, mean_q4_3rd)
# plt.xlabel('log of n (half size of the graph)')
# plt.title('E[1/|lambda_M(B_n)|^3]')
# plt.savefig('./plots/plot_lMBinv_3rd.pdf')
# plt.show()

# plt.plot(np.arange(12)+2, mean_q4_6th)
# plt.xlabel('log of n (half size of the graph)')
# plt.title('E[1/|lambda_M(B_n)|^6] (quantity 4)')
# plt.savefig('./plots/plot_lMBinv_6th.pdf')
# plt.show()

# plt.plot(np.arange(12)+2, np.log(mean_q4_2nd))
# plt.xlabel('log of n (half size of the graph)')
# plt.title('log of E[1/|lambda_M(B_n)|^2]')
# plt.savefig('./plots/plot_lMBinv_2nd_log.pdf')
# plt.show()

# plt.plot(np.arange(12)+2, np.log(mean_q4_3rd))
# plt.xlabel('log of n (half size of the graph)')
# plt.title('log of E[1/|lambda_M(B_n)|^3]')
# plt.savefig('./plots/plot_lMBinv_3rd_log.pdf')
# plt.show()

# plt.plot(np.arange(12)+2, np.log(mean_q4_6th))
# plt.xlabel('log of n (half size of the graph)')
# plt.title('log of E[1/|lambda_M(B_n)|^6] (quantity 4)')
# plt.savefig('./plots/plot_lMBinv_6th_log.pdf')
# plt.show()
