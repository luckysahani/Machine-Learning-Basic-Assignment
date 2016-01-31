#!/bin/bash

printf "\n\nPart C \n-------------------------------------------------\n
We will consider 10 cases in which, for each case we will take some folder as a testing folder 
and other folders as training folders.The accuracy for each of the 10 cases are given below:\n"

for i in {1..10}
do
   printf "\nCase $i: Testing folder is $i:\n-------------------------------------------------\n"
   python 1c.py $i | awk '/---------/{y=1;next}y'
done

