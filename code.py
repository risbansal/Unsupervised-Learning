import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
import operator
import warnings
import argparse

start = time.time()
warnings.filterwarnings("error")
class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.order = []
    def addEdge(self,u,v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    def DFS(self,v,vertex):
        visited = [False]*vertex
        self.DFSUtil(v,visited)

    def DFSUtil(self,v,visited):
        visited[v]=True
        self.order.append(v)

        for i in self.graph[v]:
            if visited[i] == False:
                self.DFSUtil(i,visited)

def chow_liu(train1, valid, test):
    train = np.concatenate((train1, valid), axis=0)
    prob1 = train.sum(axis=0)
    prob1 = prob1 + 1
    prob0 = train.shape[0] - prob1 + 2
    prob1 = np.true_divide(prob1, train.shape[0] + 2)
    prob0 = np.true_divide(prob0, train.shape[0] + 2)
    weights = np.zeros((train.shape[1], train.shape[1]))
    for col1 in range(int(train.shape[1])):
        for col2 in range(int(train.shape[1])):
            if col1 == col2:
                pass
            else:
                a00 = (train[:, col1] == 0) & (train[:, col2] == 0)
                a01 = (train[:, col1] == 0) & (train[:, col2] == 1)
                a10 = (train[:, col1] == 1) & (train[:, col2] == 0)
                a11 = (train[:, col1] == 1) & (train[:, col2] == 1)
                b00 = a00.sum()
                b01 = a01.sum()
                b10 = a10.sum()
                b11 = a11.sum()
                P00 = np.true_divide(b00 + 1, int(train.shape[0]) + 4)
                P01 = np.true_divide(b01 + 1, int(train.shape[0]) + 4)
                P10 = np.true_divide(b10 + 1, int(train.shape[0]) + 4)
                P11 = np.true_divide(b11 + 1, int(train.shape[0]) + 4)
                W = P00 * np.log(np.true_divide(P00, (prob0[col1] * prob0[col2])))
                W = W + P01 * np.log(np.true_divide(P01, (prob0[col1] * prob1[col2])))
                W = W + P10 * np.log(np.true_divide(P10, (prob1[col1] * prob0[col2])))
                W = W + P11 * np.log(np.true_divide(P11, (prob1[col1] * prob1[col2])))
                weights[col1][col2] = W
    weights = weights * (-1)
    e = csr_matrix(weights)
    f = minimum_spanning_tree(e).toarray()
    f_csr = csr_matrix(f)
    l1, l2 = f_csr.toarray().nonzero()
    edges = zip(l1, l2)
    graph = Graph()
    for e in edges:
        graph.addEdge(e[0], e[1])

    graph.DFS(0, weights.shape[0])
    o = graph.order
    pa = {o[0]: np.nan}
    for i in range(1, len(o)):
        if o[i] in graph.graph[o[i - 1]]:
            pa[o[i]] = o[i - 1]
        else:
            for j in range(i - 1):
                if o[i] in graph.graph[o[i - j - 2]]:
                    pa[o[i]] = o[i - j - 2]
                    break
                else:
                    pass

    cpt_mat = []
    for child in list(pa.keys())[1:]:
        A00 = (train[:, child] == 0) & (train[:, pa[child]] == 0)
        A01 = (train[:, child] == 1) & (train[:, pa[child]] == 0)
        A10 = (train[:, child] == 0) & (train[:, pa[child]] == 1)
        A11 = (train[:, child] == 1) & (train[:, pa[child]] == 1)
        B00 = A00.sum()
        B01 = A01.sum()
        B10 = A10.sum()
        B11 = A11.sum()
        p00 = np.true_divide(B00 + 1, (train[:, pa[child]] == 0).sum() + 2)
        p01 = np.true_divide(B01 + 1, (train[:, pa[child]] == 0).sum() + 2)
        p10 = np.true_divide(B10 + 1, (train[:, pa[child]] == 1).sum() + 2)
        p11 = np.true_divide(B11 + 1, (train[:, pa[child]] == 1).sum() + 2)
        cpt_mat.append([p00, p01, p10, p11])
    cpt_mat = np.array(cpt_mat)
    lcpt_mat = np.log(cpt_mat)

    lpprobX0 = np.log(np.true_divide((train[:, 0].sum() + 1), train.shape[0] + 2))
    lnprobX0 = np.log(np.true_divide((train.shape[0] - train[:, 0].sum() + 1), train.shape[0] + 2))

    t = test.copy()
    t = t.astype(float)
    for i in range(1, train.shape[1]):
        par = pa[i]
        t[(test[:, i] == 0) & (test[:, par] == 0), i] = lcpt_mat[i - 1][0]
        t[(test[:, i] == 1) & (test[:, par] == 0), i] = lcpt_mat[i - 1][1]
        t[(test[:, i] == 0) & (test[:, par] == 1), i] = lcpt_mat[i - 1][2]
        t[(test[:, i] == 1) & (test[:, par] == 1), i] = lcpt_mat[i - 1][3]

    t[(test[:, 0]) == 0, 0] = lnprobX0
    t[(test[:, 0]) == 1, 0] = lpprobX0

    LLH_col = t.sum(axis=1)
    mLL = LLH_col.mean()
    return mLL


def independent(train1, valid, test):
    train = np.concatenate((train1, valid), axis = 0)
    prob1 = train.sum(axis=0)
    prob0 = train.shape[0] - prob1
    prob1 = prob1 + 1
    prob0 = prob0 + 1
    lprob1 = np.log(np.true_divide(prob1, train.shape[0] + 2))
    lprob0 =  np.log(np.true_divide(prob0, train.shape[0] + 2))
    test1 = (test * lprob1).astype(float)
    test2 = np.ma.array(test1, mask=test1 != 0)
    test3 = test2 + lprob0
    test4 = test3.data.sum(axis=1)
    test5 = test4.mean()
    return test5


def RFtree(train, valid, test, od, fn):
    #nos = [3, 5, 7]
    sample_cpt_dict = {}
    tset_llh = []

    for n in range(10):
        print("evaluation no. : {}".format(n + 1))
        samples = []
   
        for k in range(od[fn][0]):
            temp_arr = np.array([np.inf] * train.shape[1])
            for i in range(int(train.shape[0] / 3)):
                ri = np.random.randint(0, train.shape[0])
                temp_arr = np.vstack((temp_arr, train[ri]))
            temp_arr = temp_arr[1:, :]
            samples.append(temp_arr)
      
        cpt_list = []
        for s in range(len(samples)):
            pprob = samples[s].sum(axis=0)
            pprob = pprob + 1
            nprob = samples[s].shape[0] - pprob + 2
            pprob = np.true_divide(pprob, samples[s].shape[0] + 2)
            nprob = np.true_divide(nprob, samples[s].shape[0] + 2)
            weights = np.zeros((samples[s].shape[1], samples[s].shape[1]))

            for col1 in range(int(samples[s].shape[1])):
                for col2 in range(int(samples[s].shape[1])):
                    if col1 == col2:
                        pass
                    else:
                        a00 = (samples[s][:, col1] == 0) & (samples[s][:, col2] == 0)
                        a01 = (samples[s][:, col1] == 0) & (samples[s][:, col2] == 1)
                        a10 = (samples[s][:, col1] == 1) & (samples[s][:, col2] == 0)
                        a11 = (samples[s][:, col1] == 1) & (samples[s][:, col2] == 1)
                        b00 = a00.sum()
                        b01 = a01.sum()
                        b10 = a10.sum()
                        b11 = a11.sum()
                        P00 = np.true_divide(b00 + 1, int(samples[s].shape[0]) + 4)
                        P01 = np.true_divide(b01 + 1, int(samples[s].shape[0]) + 4)
                        P10 = np.true_divide(b10 + 1, int(samples[s].shape[0]) + 4)
                        P11 = np.true_divide(b11 + 1, int(samples[s].shape[0]) + 4)
                        W = P00 * np.log(np.true_divide(P00, (nprob[col1] * nprob[col2])))
                        W = W + P01 * np.log(np.true_divide(P01, (nprob[col1] * pprob[col2])))
                        W = W + P10 * np.log(np.true_divide(P10, (pprob[col1] * nprob[col2])))
                        W = W + P11 * np.log(np.true_divide(P11, (pprob[col1] * pprob[col2])))
                        weights[col1][col2] = W
            for num in range(od[fn][1]):
                ri1 = np.random.randint(0, train.shape[1])
                ri2 = np.random.randint(0, train.shape[1])
                weights[ri1][ri2] = 0
                weights[ri2][ri1] = 0
            weights = weights * (-1)
            e = csr_matrix(weights)
            f = minimum_spanning_tree(e).toarray()
            f_csr = csr_matrix(f)
            l1, l2 = f_csr.toarray().nonzero()
            edges = zip(l1, l2)
            graph = Graph()
            for e in edges:
                graph.addEdge(e[0], e[1])
            # The 0th feature is chosen as the root node
            graph.DFS(0, weights.shape[0])
            o = graph.order
            pa = {o[0]: np.nan}
            for i in range(1, len(o)):
                if o[i] in graph.graph[o[i - 1]]:
                    pa[o[i]] = o[i - 1]
                else:
                    for j in range(i - 1):
                        if o[i] in graph.graph[o[i - j - 2]]:
                            pa[o[i]] = o[i - j - 2]
                            break
                        else:
                            pass

            cpt_mat = []
            for child in list(pa.keys())[1:]:
                A00 = (samples[s][:, child] == 0) & (samples[s][:, pa[child]] == 0)
                A01 = (samples[s][:, child] == 1) & (samples[s][:, pa[child]] == 0)
                A10 = (samples[s][:, child] == 0) & (samples[s][:, pa[child]] == 1)
                A11 = (samples[s][:, child] == 1) & (samples[s][:, pa[child]] == 1)
                B00 = A00.sum()
                B01 = A01.sum()
                B10 = A10.sum()
                B11 = A11.sum()
                # temp1 = (samples[s][:, pa[child]] == 0).sum()
                # temp2 = (samples[s][:, pa[child]] == 1).sum()
                p00 = np.true_divide(B00 + 1, (samples[s][:, pa[child]] == 0).sum() + 2)
                p01 = np.true_divide(B01 + 1, (samples[s][:, pa[child]] == 0).sum() + 2)
                p10 = np.true_divide(B10 + 1, (samples[s][:, pa[child]] == 1).sum() + 2)
                p11 = np.true_divide(B11 + 1, (samples[s][:, pa[child]] == 1).sum() + 2)
                cpt_mat.append([p00, p01, p10, p11])
            cpt_mat = np.array(cpt_mat)
            lcpt_mat = np.log(cpt_mat)
            # log of probabilities of the first feature which is the root
            lpprobX0 = np.log(np.true_divide((samples[s][:, 0].sum() + 1), samples[s].shape[0] + 2))
            lnprobX0 = np.log(
                np.true_divide((samples[s].shape[0] - samples[s][:, 0].sum() + 1), samples[s].shape[0] + 2))
            lcpt_mat = np.vstack(([lpprobX0, lnprobX0, 0, 0], lcpt_mat))
            cpt_list.append(lcpt_mat)
        sample_cpt_dict[od[fn][0]] = cpt_list

        f_llh_sum = []

        for num in range(od[fn][0]):
            t = tsmat.copy()
            t = t.astype(float)
            for i in range(1, tsmat.shape[1]):
                par = pa[i]
                t[(test[:, i] == 0) & (test[:, par] == 0), i] = sample_cpt_dict[od[fn][0]][num][i][0]
                t[(test[:, i] == 1) & (test[:, par] == 0), i] = sample_cpt_dict[od[fn][0]][num][i][1]
                t[(test[:, i] == 0) & (test[:, par] == 1), i] = sample_cpt_dict[od[fn][0]][num][i][2]
                t[(test[:, i] == 1) & (test[:, par] == 1), i] = sample_cpt_dict[od[fn][0]][num][i][3]
            # filling up the 0th column in t with the respective probabilities
            t[(test[:, 0]) == 0, 0] = sample_cpt_dict[od[fn][0]][num][0][1]
            t[(test[:, 0]) == 1, 0] = sample_cpt_dict[od[fn][0]][num][0][0]
            llh = t.sum(axis=1)
            llh = llh.mean()
        f_llh_sum.append(np.true_divide(llh, od[fn][0]))
        f_llh_sum = np.array(f_llh_sum)
        f_llh = f_llh_sum.sum()
        tset_llh.append(f_llh)
    tset_llh = np.array(tset_llh)
    t1 = tset_llh.mean()
    t2 = tset_llh.std()
    return tset_llh, t1, t2


def mixed_trees(train1, valid, test, od, fn):
    train = train1.copy()
    K_selection = {}
    K_cpts = {}
    K_lambdas = {}
    tset_llh = []
   

        for i in range(5):
        test_llh_sum = []
        print("{}th evaluation".format(i))
        
        h_mat = np.random.rand(train.shape[0], od[fn])
        hmat_l = []
        llhs_list = []
        cpts_l = []
       
        epochs = 5
        for i in range(epochs):
            if (i + 1) % 10 == 0:
                print("epoch no. : {}".format(i + 1))
            h_mat = h_mat / h_mat.sum(axis=1)[:, None]
           
            hmat_l.append(h_mat.copy())
            llh_list = []
            cpt_l = []
            for k in range(od[fn]):
                l = np.true_divide(h_mat[:, k].sum(), h_mat.shape[0])
                weights = np.zeros((train.shape[1], train.shape[1]))
                for col1 in range(int(train.shape[1])):
                    for col2 in range(int(train.shape[1])):
                        if col1 == col2:
                            pass
                        else:
                            a00 = h_mat[:, k][(train[:, col1] == 0) & (train[:, col2] == 0)]
                            a01 = h_mat[:, k][(train[:, col1] == 0) & (train[:, col2] == 1)]
                            a10 = h_mat[:, k][(train[:, col1] == 1) & (train[:, col2] == 0)]
                            a11 = h_mat[:, k][(train[:, col1] == 1) & (train[:, col2] == 1)]
                            b00 = a00.sum()
                            b01 = a01.sum()
                            b10 = a10.sum()
                            b11 = a11.sum()

                            P00 = np.true_divide(b00 + l, (h_mat[:, k].sum() + 4 * l))
                            P01 = np.true_divide(b01 + l, (h_mat[:, k].sum() + 4 * l))
                            P10 = np.true_divide(b10 + l, (h_mat[:, k].sum() + 4 * l))
                            P11 = np.true_divide(b11 + l, (h_mat[:, k].sum() + 4 * l))
                            
                            pprob0 = h_mat[:, k][train[:, col1] == 1]
                            pprob0 = np.true_divide(pprob0.sum() + l, h_mat[:, k].sum() + 2 * l)
                            pprob1 = h_mat[:, k][train[:, col2] == 1]
                            pprob1 = np.true_divide(pprob1.sum() + l, h_mat[:, k].sum() + 2 * l)
                            nprob0 = h_mat[:, k][train[:, col1] == 0]
                            nprob0 = np.true_divide(nprob0.sum() + l, h_mat[:, k].sum() + 2 * l)
                            nprob1 = h_mat[:, k][train[:, col2] == 0]
                            nprob1 = np.true_divide(nprob1.sum() + l, h_mat[:, k].sum() + 2 * l)

                            W = P00 * np.log(np.true_divide(P00, (nprob0 * nprob1)))
                            W = W + P01 * np.log(np.true_divide(P01, (nprob0 * pprob1)))
                            W = W + P10 * np.log(np.true_divide(P10, (pprob0 * nprob1)))
                            W = W + P11 * np.log(np.true_divide(P11, (pprob0 * pprob1)))
                            weights[col1][col2] = W
                weights = weights * (-1)
                e = csr_matrix(weights)
                f = minimum_spanning_tree(e).toarray().astype(float)
                f_csr = csr_matrix(f)
                l1, l2 = f_csr.toarray().nonzero()
                edges = zip(l1, l2)
                graph = Graph()
                for e in edges:
                    graph.addEdge(e[0], e[1])
                
                graph.DFS(0, weights.shape[0])
                o = graph.order
                pa = {o[0]: np.nan}
                for i in range(1, len(o)):
                    if o[i] in graph.graph[o[i - 1]]:
                        pa[o[i]] = o[i - 1]
                    else:
                        for j in range(i - 1):
                            if o[i] in graph.graph[o[i - j - 2]]:
                                pa[o[i]] = o[i - j - 2]
                                break
                            else:
                                pass

                cpt_mat = []

                for child in list(pa.keys())[1:]:
                    try:
                        lc0 = np.true_divide((h_mat[:, k][train[:, pa[child]] == 0]).sum(), (train[:, pa[child]] == 0).sum())
                    except RuntimeWarning:
                        lc0 = 0.5
                    try:
                        lc1 = np.true_divide(h_mat[:, k][train[:, pa[child]] == 1].sum(), (train[:, pa[child]] == 1).sum())
                    except RuntimeWarning:
                        lc1 = 0.5
       
                    A00 = h_mat[:, k][(train[:, child] == 0) & (train[:, pa[child]] == 0)]
                    A00 = np.true_divide(A00.sum() + lc0, h_mat[:, k][train[:, pa[child]] == 0].sum() + 2 * lc0)
                    A01 = h_mat[:, k][(train[:, child] == 1) & (train[:, pa[child]] == 0)]
                    A01 = np.true_divide(A01.sum() + lc0, h_mat[:, k][train[:, pa[child]] == 0].sum() + 2 * lc0)
                    A10 = h_mat[:, k][(train[:, child] == 0) & (train[:, pa[child]] == 1)]
                    A10 = np.true_divide(A10.sum() + lc1, h_mat[:, k][train[:, pa[child]] == 1].sum() + 2 * lc1)
                    A11 = h_mat[:, k][(train[:, child] == 1) & (train[:, pa[child]] == 1)]
                    A11 = np.true_divide(A11.sum() + lc1, h_mat[:, k][train[:, pa[child]] == 1].sum() + 2 * lc1)
                    cpt_mat.append([A00, A01, A10, A11])

                cpt_mat = np.array(cpt_mat)
                lpprobX0 = h_mat[:, k][(train[:, 0] == 1)]
                lpprobX0 = np.log(np.true_divide(lpprobX0.sum() + l, h_mat[:, k].sum() + 2 * l))

                lnprobX0 = h_mat[:, k][(train[:, 0] == 0)]
                lnprobX0 = np.log(np.true_divide(lnprobX0.sum() + l, h_mat[:, k].sum() + 2 * l))

                lcpt_mat = np.log(cpt_mat)
                lcpt_mat = np.vstack((np.array([lpprobX0, lnprobX0, 0, 0]), lcpt_mat))
                cpt_l.append(lcpt_mat)

                t = train.copy()
                t = t.astype(float)
                for i in range(1, train.shape[1]):
                    par = pa[i]
                    t[(train[:, i] == 0) & (train[:, par] == 0), i] = lcpt_mat[i][0]
                    t[(train[:, i] == 1) & (train[:, par] == 0), i] = lcpt_mat[i][1]
                    t[(train[:, i] == 0) & (train[:, par] == 1), i] = lcpt_mat[i][2]
                    t[(train[:, i] == 1) & (train[:, par] == 1), i] = lcpt_mat[i][3]
                
                t[(train[:, 0]) == 0, 0] = lcpt_mat[0][1]
                t[(train[:, 0]) == 1, 0] = lcpt_mat[0][0]
                
                LLH_col = t.sum(axis=1)
                llh_list.append(LLH_col.mean())
                Phgx = h_mat[:, k] * np.exp(LLH_col)
                if i == epochs - 1:
                    pass
                else:
                    h_mat[:, k] = Phgx
            cpts_l.append(cpt_l)
            llhs_list.append(llh_list)
            K_cpts[od[fn]] = cpts_l[-1]

            temp_l = [np.true_divide(h_mat[:, hcol].sum(), h_mat.shape[0]) for hcol in range(od[fn])]

            K_lambdas[od[fn]] = temp_l
        for k in range(od[fn]):
            
            t = test.copy()
            t = t.astype(float)
            for i in range(1, test.shape[1]):
                par = pa[i]
                t[(test[:, i] == 0) & (test[:, par] == 0), i] = K_cpts[od[fn]][k][i][0]
                t[(test[:, i] == 1) & (test[:, par] == 0), i] = K_cpts[od[fn]][k][i][1]
                t[(test[:, i] == 0) & (test[:, par] == 1), i] = K_cpts[od[fn]][k][i][2]
                t[(test[:, i] == 1) & (test[:, par] == 1), i] = K_cpts[od[fn]][k][i][3]
            
            t[(test[:, 0]) == 0, 0] = K_cpts[od[fn]][k][0][1]
            t[(test[:, 0]) == 1, 0] = K_cpts[od[fn]][k][0][0]
            llh = t.sum(axis = 1)
            llh = llh.mean()
            test_llh_sum.append(llh * K_lambdas[od[fn]][k])
        test_llh_sum = np.array(test_llh_sum)
        test_mllh = test_llh_sum.sum()
        tset_llh.append(test_mllh)
    tsetllh = np.array(tset_llh)
    t1 = np.mean(tset_llh)
    t2 = np.std(tset_llh)

    return tsetllh, t1, t2



#loading datasets

def main():
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-data_path', '--path', type=str)
	parser.add_argument('-dataset_name', '--dname', type=str)
	parser.add_argument('-algorithm', '--algo', type=int)


	arg = parser.parse_args()

	data_path = arg.path
	dataset = arg.dname
	algos = arg.algo


	train_data = np.loadtxt(data_path + '/' + dataset + ".ts.data", delimiter=',')
	valid_data = np.loadtxt(data_path + '/' + dataset + ".valid.data", delimiter=',')
	test_data = np.loadtxt(data_path + '/' + dataset + ".test.data", delimiter=',')


	if(algos == 1):
		#Independent bayesian networks
		ind_ll = independent(train_data, valid_data, test_data)
		print("Dataset name = {}".format(dataset))
		print("Average log-likelihood for independent bayesian networks : ", ind_ll)

	if(algos == 2):
		#Chow-liu tree
		cl_ll = chow_liu(train_data, valid_data, test_data)
		print("Dataset Name = {}".format(dataset))
		print("Average log-likelihood for chow_liu : ", cl_ll)

	if(algos == 3):
		#Mixture of trees
		print("Dataset Name = {}".format(dataset))
		optimal_k = {"accidents" : 2, "baudio" : 4, "bnetflix" : 3, "dna" : 4, "jester" : 2, "kdd" : 4, "msnbc" : 3, "nltcs" : 2, "plants" : 4, "r52" : 2}
		mixed_ll, mixed_avg, mixed_std_dev = mixed_trees(train_data, valid_data, test_data, optimal_k, dataset)
		print("loglikelihoods : {}".format(mixed_ll))
		print("Average: {}".format(mixed_avg))
		print("Standard deviation: {}".format(mixed_std_dev))

	if(algos == 4):
		#random forest tree
		print("Dataset Name = {}".format(dataset))
		optimal_kr = {"accidents": [2, 6], "baudio": [4, 5], "bnetflix": [3, 7], "dna": [3, 6], "jester": [4, 7],"kdd": [2, 5], "msnbc": [2, 7], "nltcs": [2, 6], "plants": [2, 7], "r52": [4, 5]}
		rf_ll, rf_avg, rf_sd = RFtree(train_data, valid_data, test_data, optimal_kr, dataset)
		print("likelihood {}".format(rv_ll))
		print("Average loglikelihood of the data using RFtree = {}".format(rf_avg)
		print("Standard deviation = {}".format(rf_sd))





	print ("Execution Time",time.time() - start)


main()