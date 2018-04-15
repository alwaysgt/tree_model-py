import numpy as np
import math
from pdb import set_trace

class base_tree:
	class node:
		def __init__(self):
			# range is a binary tuple, the data stay in [ range[0],range[1] [
			self.range = None 
			# value is the best prediction for points in this node when it is a leaf.
			self.value = None
			# distance from the root
			self.depth = 0
			# whether it is a leave
			self.is_leaf = False 

			# if it is not a leaf, it has following attribute
			self.feature = None	
			self.threshold = None
			self.left_child = None
			self.right_child = None 
			# improvement is the amount of improvement it get for the risk function
			self.improvement = None 
					
			

	def __init__(self,max_depth = 2):
		# sample is the rearrange index.
		# sample[i] is the index of row in X of the i-sample
		# sample is a new coordinate! X_index is also a coordinate!
		self.sample = None 
		# stopping condition
		self.min_splitting = 2
		self.max_depth = max_depth 
		

	def fit(self,X,Y):
		self.n_sample,self.n_features  = X.shape

		# checking the shape
		if self.n_sample != len(Y):
			raise ValueError('Size of X and Y don\'t match!')


		# self.sample is a new new coordinate of data
		# subjected to updating
		self.sample = list(range(self.n_sample))

		# saving the 
		self.X = X.copy()
		self.Y = Y.copy()

		

		#find the index of samples under which any column of X is increasing
		# Usage: X[X_index[i,j],:] for i-th smallest sample wrt the j-th feature
		self.X_index = np.argsort(X,axis = 0)

		#adding root to the stack
		stack = []
		self.root = self.node()
		self.root.range = (0,self.n_sample)
		stack.append(self.root)

		#finding the partition, using a stack to go through all the noe 
		while stack:
			current_node = stack.pop()		

			# return left node, right node, feature, threshold 
			# And assign to the current node
			# Could be convient if use side effect to change sample
			# return the child if not a leaf
			temp = self.splitting(X,Y,self.sample,self.X_index,current_node)


			if temp:
				for i in temp:
					stack.append(i)

				

	def predict(self,X):
		def predict_1d(x):
			# predict for a given row
			current_node = self.root
			while not current_node.is_leaf:
				feature = current_node.feature
				threshold = current_node.threshold
				if x[feature] > threshold:
					current_node = current_node.right_child
				else:
					current_node = current_node.left_child
			return current_node.value

		return np.array([predict_1d(X[i,:]) for i in range(X.shape[0])])

	def print(self):
		stack  =[self.root]
		while stack:
			temp = stack.pop()
			data = [(list(self.X[self.sample[i],:]),self.Y[self.sample[i]]) for i in range(*temp.range) ]
			temp_ = "" + ("    |")*temp.depth
			print((temp_+"\n")*4)
			# data.sort()
			print(temp_+ "----",data)
			if not temp.is_leaf:
				print(temp_+ "----","feature:{},and threshold:{} ".format(temp.feature,temp.threshold))

			if not temp.is_leaf:
				stack.append(temp.right_child)
				stack.append(temp.left_child)






class regression_tree(base_tree):
	def __init__(self,**kw):
		super(regression_tree,self).__init__(**kw)

	# implement of splitting in regression
	def splitting(self,X,Y,sample,X_index, current_node):
		# Our goal is to minimize the error, so let the starting value be infty
		current_range = current_node.range
		n_subsample = current_range[1]-current_range[0]
		best_feature = None
		best_threshold = None
		best_value = math.inf
		

		# mask the data 
		# mask[i] == 1 if the i-th row in in sample
		mask = np.zeros(self.n_sample, dtype= bool)
		sum_of_response = 0
		# i is the index of sample. 
		for i in range(*current_range):
			mask[sample[i]] = True
			sum_of_response = sum_of_response  + Y[sample[i]]

		# a wrong one
		# for i in range(n_sample):
		# 	if sample[i] >= current_range[0] and sample[i] < current_range[1]:
		# 		mask[i] = True
		# 		sum_of_response = sum_of_response + y[i]


		# Calculating the best predict for this node 
		current_node.value = sum_of_response / n_subsample

		# decide wether return is_leaf:
		if n_subsample < self.min_splitting  or current_node.depth >= self.max_depth:
			current_node.is_leaf = True
			return None



		# find the best partition
		for i in range(self.n_features):
			current_left = 0
			current_right = sum_of_response
			current_feature = i
			f_index = X_index[:,i]

			# j is ordered index wrt to the i-th feature. 
			# Get the best feature best value and best threshold
			# Be careful to tie! Always find one more element.
			# n_pass is the 1 + index of data we are dealting 
			n_pass = 0
			for j in range(self.n_sample):
				if mask[f_index[j]] == 1:
					n_pass = n_pass +1
					if n_pass == 1:
						last_index = f_index[j]
						continue
					now_index = f_index[j]
					current_left = current_left + Y[last_index]
					current_right = current_right - Y[last_index]

					if X[now_index,i] == X[last_index,i]:
						continue
					# set_trace()
					# if n_pass == n_subsample:
					# 	current_value = - current_left**2/(n_pass-1)
					# else:
					current_value = - current_left**2/(n_pass-1)  - current_right**2/(n_subsample - (n_pass-1))
					if current_value < best_value:
						best_value = current_value
						# the best it should be (-\infty, threshold] (threshold,\infty)
						best_threshold = X[last_index,i]/2.0 + X[now_index,i]/2.0 
						best_feature = i
					if current_range == (0,3):
						# set_trace()
						pass
					last_index = now_index


		# reordering the data
		# endpoint belongs to left
		end = current_range[1] - 1
		start = current_range[0]
		while start <= end:
			if X[sample[start],best_feature] > best_threshold:
				# exchange it with the end
				temp =sample[start]
				sample[start] = sample[end]
				sample[end] = temp
				end = end -1
			else:
				start = start + 1

		current_node.feature = best_feature
		current_node.threshold = best_threshold
		current_node.improvement = sum_of_response**2/n_subsample + best_value





		left_child = self.node()
		left_child.range = (current_range[0],end+1)
		left_child.depth = current_node.depth + 1
		current_node.left_child = left_child
		right_child = self.node()
		right_child.range =(end+1,current_range[1])
		right_child.depth = current_node.depth + 1
		current_node.right_child = right_child

		# Return  left_child, right_child, feature, threshold, is_leaf, value
		return left_child,right_child
 
	def printnode(self,node):
		print("\n")
		for i in range(*node.range):
			print(Y[self.sample[i]])
		print("\n")


if __name__ == "__main__":
	X = np.array([[4,2,3,4,5,5],[3,2,1,4,5,2]]).transpose()
	Y = np.array([100,2,2,1,1,2])
	rt = regression_tree()
	rt.fit(X,Y)
	rt.print()
	print("\n\n\n")
	print(rt.predict(X))
	print(rt.sample)


