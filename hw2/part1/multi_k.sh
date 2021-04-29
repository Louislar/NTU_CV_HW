#!/bin/bash
#######s program shows "Hello World!" in your screen.
# History:
# 2015/07/16	VBird	First release#####
echo -e "Start trying multiple k in KNN.With same Sift BoW parameters."
max_try_k=100
echo -e ${max_try_k}
for (( i=1; i<=10; i=i+1 ))
do
	echo -e ${i}
	python p1.py --feature sift_bag_of_words --classifier knn 
done
echo -e "End"
