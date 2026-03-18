# Simple Makefile for Gemmulem (alternative to CMake)
# Usage: make && sudo make install

CC      ?= gcc
CFLAGS  ?= -O2 -march=native -Wall -Wextra
PREFIX  ?= /usr/local
LDFLAGS ?= -lm

# Auto-detect OpenMP
OMPFLAG := $(shell echo 'int main(){}' | $(CC) -fopenmp -x c - -o /dev/null 2>/dev/null && echo -fopenmp)

SRC_DIR  = src/lib
SOURCES  = $(SRC_DIR)/EM.c $(SRC_DIR)/distributions.c $(SRC_DIR)/pearson.c \
           $(SRC_DIR)/multivariate.c $(SRC_DIR)/streaming.c $(SRC_DIR)/simd_estep.c
HEADERS  = $(wildcard $(SRC_DIR)/*.h)
OBJECTS  = $(SOURCES:.c=.o)

.PHONY: all clean install test

all: gemmulem

$(SRC_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS)
	$(CC) $(CFLAGS) $(OMPFLAG) -mavx2 -c -o $@ $<

gemmulem: src/main.cpp $(OBJECTS)
	$(CXX) $(CFLAGS) $(OMPFLAG) -I$(SRC_DIR) -o $@ src/main.cpp $(OBJECTS) $(LDFLAGS)

clean:
	rm -f $(OBJECTS) gemmulem

install: gemmulem
	install -d $(PREFIX)/bin $(PREFIX)/lib $(PREFIX)/include/gemmulem
	install -m 755 gemmulem $(PREFIX)/bin/
	install -m 644 $(HEADERS) $(PREFIX)/include/gemmulem/

test: gemmulem
	@echo "Quick smoke test..."
	@echo "1 2 3 4 5 6 7 8 9 10" | tr ' ' '\n' > /tmp/gem_test.txt
	@./gemmulem -g /tmp/gem_test.txt -d Gaussian -k 2 -o /tmp/gem_test_out.csv && echo "✅ PASS" || echo "❌ FAIL"
	@rm -f /tmp/gem_test.txt /tmp/gem_test_out.csv
