CC=gcc
CFLAGS=-std=gnu11 -Wall -Wextra -m64 -O0 -g -pg
LDFLAGS=-lm

SRC=$(wildcard src/*.c)
OBJ=$(patsubst src/%.c,obj/%.o,$(SRC))

all: $(OBJ)
	mkdir -p bin/
	$(CC) $(LDFLAGS) $(OBJ) -o bin/neuralnet

obj/%.o: src/%.c
	mkdir -p obj/
	$(CC) $(CFLAGS) -c $< -o $@

test:
	echo -e "\nCC=$(CC)\nCFLAGS=$(CFLAGS)\nLDFLAGS=$(LDFLAGS)\nSRC=$(SRC)\nOBJ=$(OBJ)\n"

clean:
	rm -rfv bin/* obj/*
