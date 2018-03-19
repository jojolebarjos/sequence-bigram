
all:
	g++ -o bigram bigram.cpp -Ofast -fopenmp -Wall -Wextra -std=c++11 -pedantic
