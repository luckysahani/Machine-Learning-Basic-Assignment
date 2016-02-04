#!/bin/bash

printf "Question 1\n=================================================\n" > 1_result.txt
./1a.sh > 1a_result.txt
cat 1a_result.txt >> 1_result.txt
./1b.sh > 1b_result.txt
cat 1b_result.txt >> 1_result.txt
./1c.sh > 1c_result.txt
cat 1c_result.txt >> 1_result.txt

