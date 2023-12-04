OBJECTS= constructs-nikos.o 
CPPFLAGS=-g3 -pthread -Wall 


all: lamodel-nikos tests

runs:
	./lamodel-nikos -o numPatterns=20 -o hasCaap=0 &
	./lamodel-nikos -o numPatterns=20 -o hasCaap=1 &

vis:
	python3 engrams.py

constructs-nikos.o: constructs-nikos.h

lamodel-nikos: $(OBJECTS) lamodel-nikos.o 
	g++  $(OBJECTS) lamodel-nikos.o -o lamodel-nikos -lm    -L/usr/local/lib/ -std=c++11 # -lGLU -lglut 


remake: clean all

cleanup:
	rm -f ../submit_lamodel.sh.e*
	rm -f ../submit_lamodel.sh.o*

clean: cleanup
	rm -f *.o lamodel-nikos.o*  ../lamodel-nikos ../tests

cdata:
	sh run_cluster.sh

data:
	sh run_serial.sh


