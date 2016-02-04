#!/bin/bash

printf "Question 2\n=================================================\n" > 2_result.txt
./2a.sh > 2a_result.txt
cat 2a_result.txt >> 2_result.txt
./2b.sh > 2b_result.txt
cat 2b_result.txt >> 2_result.txt
./2c.sh > 2c_result.txt
cat 2c_result.txt >> 2_result.txt

