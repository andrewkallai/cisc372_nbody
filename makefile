FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile
COMPILER=gcc
EXT=c

ifeq ($(CC),nvcc)
	COMPILER=nvcc
	EXT=cu
endif

nbody: nbody.o compute.o
	$(COMPILER) $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.$(EXT) planets.h config.h vector.h $(ALWAYS_REBUILD)
	$(COMPILER) $(FLAGS) -c $< 
compute.o: compute.$(EXT) config.h vector.h $(ALWAYS_REBUILD)
	$(COMPILER) $(FLAGS) -c $< 
clean:
	rm -f *.o nbody 
