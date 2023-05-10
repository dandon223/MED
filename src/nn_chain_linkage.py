import numpy as np
import time
import math

class NNChainLinkage():

    def __init__(self, algorithm='euclidean', formula='single'):
        self.pairwise_diss_dict = {
            'euclidean': self._euclidean,
            'manhattan': self._manhattan
        }
        self.formula_dict = {
            'single': self._single,
            'complete': self._complete,
            'average': self._average
        }
        if algorithm not in self.pairwise_diss_dict:
            print("---Available metrics---")
            print(list(self.pairwise_diss_dict.keys()))
            raise Exception(
                f"Unrecognized distance algorithm: {algorithm}.")
        self.pairwise_diss = self.pairwise_diss_dict[algorithm]
        if formula not in self.formula_dict:
            print("---Available formulas---")
            print(list(self.pairwise_diss_dict.keys()))
            raise Exception(
                f"Unrecognized formula: {formula}.")
        self.formula = self.formula_dict[formula]

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        return self._nn_chain_linkage(data, self.pairwise_diss(data))
    
    def _nn_chain_linkage(self, data: np.ndarray, pairwise_diss: np.ndarray):

        linkage = self._nn_chain_core(len(data), pairwise_diss)
        order = np.argsort(linkage[:, 2], kind='stable')
        linkage = linkage[order]
        #print("L", linkage)
        L_prim = self._label(linkage)
        #print("L_prim", L_prim)
        return L_prim
    
    def _label(self, sorted_linkage):
        labels = np.full((len(sorted_linkage), 3), np.inf)
        union_find = UnionFind(len(sorted_linkage) + 1)

        for i in range(len(sorted_linkage)):
            a, b = int(sorted_linkage[i, 0]), int(sorted_linkage[i, 1])
            root_a = union_find.efficient_find(a)
            root_b = union_find.efficient_find(b)

            labels[i, 0] = root_a
            labels[i, 1] = root_b
            labels[i, 2] = sorted_linkage[i, 2]
            union_find.union(root_a, root_b)

        return labels

    def _nn_chain_core(self, data_size: int, pairwise_diss: np.ndarray):
        L = np.empty((data_size-1, 3))
        i=0
        S = np.arange(data_size, dtype=np.int_)
        chain = []
        size = np.full(data_size, fill_value=1)

        while len(S) > 1:
            #time.sleep(0.05)
            if len(chain) <= 3:
                #print("len(chain) <= 3:")
                a = S[0]
                b = S[1]
                chain = [a]
                #print("a, b", a, b)
                #if size[b] == 0:
                #    print("1 size[b] == 0", b)
                #    exit(-1)
            else:
                #print("else")
                #print("chain", chain)
                a = chain[-4]
                b = chain[-3]
                chain = chain[:len(chain) - 3]
                #print("chain2", chain)
                #if size[b] == 0:
                #    print("2 size[b] == 0", b)
                #    exit(-1)
            
            while True:
                #print("while True:")
                #print(a, b)
                c = b
                b_a_value = pairwise_diss[b, a]
                #print("c, b_a_value", c, b_a_value)
                S_prim = np.delete(S, np.where(S==a))
                x_a_value = pairwise_diss[[x for x in S_prim], a].min()
                #print(np.partition(pairwise_diss[[x for x in S], a], 1)[:4])
                #print("x_a_value", x_a_value)
                x_a_indices = np.where(pairwise_diss[:, a] == x_a_value)[0]
                #print("x_a_indices", x_a_indices)
                for index in x_a_indices:
                    if index in S_prim:
                        x_a_index = index
                        break

                if x_a_value < b_a_value:
                    #print("vs", x_a_value, b_a_value)
                    c = x_a_index

                #if c not in S:
                #    print("c not in S")
                #    exit(-1)

                a, b = c, a
                #print("a, b", a, b)
                #if size[b] == 0:
                #    print("3 size[b] == 0", b)
                #    exit(-1)
                chain.append(a)
                #print("chain", chain)
                if len(chain) >=3 and a == chain[-3]:
                    break

            #print("L: a, b", a, b)
            #if size[b] == 0:
            #    print("size[b] == 0", b)
            #    exit(-1)
            L[i, 0] = a
            L[i, 1] = b
            L[i, 2] = pairwise_diss[a, b]
            
            i = i + 1
            S = np.delete(S, np.where(S == a))
            S = np.delete(S, np.where(S == b))
            n = a
            size[n] = size[a] + size[b]
            size[b] = 0

            for x in S:
                diss = self.formula(pairwise_diss[a, x], pairwise_diss[b, x], pairwise_diss[a, b], size[a], size[b], size[x])
                pairwise_diss[n, x] = diss
                pairwise_diss[x, n] = diss
            S = np.append(S, n)

        #print(size)
        return L

    # single
    def _single(self, diss_a_x, diss_b_x, diss_a_b, size_a, size_b, size_x):
        #ward = (size_a+size_x) * diss_a_x + (size_b + size_x) * diss_b_x - size_x * diss_a_b
        #if ward > 1000:
        #    print(ward, size_a, size_b, size_x)
        #ward = ward/(size_a + size_b + size_x)
        #return math.sqrt(ward)
        return min(diss_a_x, diss_b_x)
    
    def _complete(self, diss_a_x, diss_b_x, diss_a_b, size_a, size_b, size_x):
        return max(diss_a_x, diss_b_x)
    
    def _average(self, diss_a_x, diss_b_x, diss_a_b, size_a, size_b, size_x):
        return (size_a * diss_a_x + size_b * diss_b_x)/(size_a + size_b)

    def _euclidean(self, data: np.ndarray) -> np.ndarray:
        n = len(data)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                distance = 0.0
                for k in range(len(data[i])):
                    distance += (data[i][k] - data[j][k])**2.0

                distance_matrix[i][j] = np.sqrt(distance)
                distance_matrix[j][i] = np.sqrt(distance)

        return distance_matrix

    def _manhattan(self, data: np.ndarray) -> np.ndarray:
        n = len(data)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                distance = 0.0
                for k in range(len(data[i])):
                    distance += abs(data[i][k] - data[j][k])

                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance

        return distance_matrix

class UnionFind:

    def __init__(self, n: int):
        self.parent = np.full(2 * n - 1, fill_value=-1)
        self.next_label = n

    def union(self, m: int, n: int):
        self.parent[m] = self.next_label
        self.parent[n] = self.next_label
        self.next_label = self.next_label + 1

    def efficient_find(self, n: int):
        p = n

        while self.parent[n] != -1:
            n = self.parent[n]

        while self.parent[p] != n:
            p, self.parent[p] = self.parent[p], n

        return n
    
def get_clusters(linkage, n_data, n_clusters):
    for link in linkage:
        if link[0] == link[1]:
            print(link)
    #print(linkage)
    clusters = {}
    curr_label = n_data
    for i in range(n_data):
        clusters[i] = [i]
    if n_clusters > n_data:
        return None
    elif n_clusters == 1:
        clusters = []
        for i in range(n_data):
            clusters.append(i)
        return [clusters]
    elif n_clusters == n_data:
        return list(clusters.values())

    i = 0
    ready = linkage.copy()
    not_ready = []
    while len(clusters) != n_clusters:
        if i == len(ready):
            i = 0
            ready = not_ready.copy()
            not_ready = []

        if ready[i][0] not in clusters.keys(
        ) or ready[i][1] not in clusters.keys():
            not_ready.append(ready[i])
            i = i + 1
            continue

        clusters[curr_label] = []
        clusters[curr_label] = [] + clusters[ready[i][0]] + clusters[ready[i][1]]
        #print(ready[i][0])
        clusters.pop(ready[i][0])
        clusters.pop(ready[i][1])
        curr_label = curr_label + 1
        i = i + 1
    return list(clusters.values())