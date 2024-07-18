#Makefile for gpufetch

# Compiler (nvcc)
CC = nvcc

# Compiler flags (none)

# Source files
SRC = gpufetch.cu

# Object files
OBJ = $(SRC:.cu=.o)

# Executable
EXE = gpufetch

# Build executable
$(EXE): $(OBJ)
	$(CC) $(OBJ) -o $(EXE)

# Build object files
%.o: %.cu
	$(CC) -c $< -o $@

# Clean
clean:
	rm -f $(OBJ) $(EXE)
