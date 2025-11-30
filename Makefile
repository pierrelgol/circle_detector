CC       = clang
CFLAGS   = -O3 -mtune=native -std=c11 -fPIC
CXX      = clang++
CXXFLAGS = -O3 -mtune=native -std=c++17 -fPIC -I/usr/include/opencv4
LDFLAGS  = -lm -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui

PREFIX ?= /usr/local

# Library
LIB_NAME = libcircleDetector
LIB_STATIC = $(LIB_NAME).a
LIB_SHARED = $(LIB_NAME).so
LIB_SRC = circleDetector.c
LIB_OBJ = $(LIB_SRC:.c=.o)

# Demos
DEMO_DIR = demo
DEMO_CV = $(DEMO_DIR)/main_cv
DEMO_SPAG = $(DEMO_DIR)/main_spag

all: $(LIB_STATIC) $(LIB_SHARED) demos

$(LIB_STATIC): $(LIB_OBJ)
	ar rcs $@ $^

$(LIB_SHARED): $(LIB_OBJ)
	$(CC) -shared -o $@ $^

demos: $(DEMO_CV) $(DEMO_SPAG)

$(DEMO_CV): $(DEMO_DIR)/main_cv.o
	$(CXX) $^ -o $@ $(LDFLAGS)

$(DEMO_SPAG): $(DEMO_DIR)/main_spag.o $(LIB_STATIC)
	$(CXX) $^ -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

install: $(LIB_STATIC) $(LIB_SHARED)
	install -d $(PREFIX)/lib
	install -d $(PREFIX)/include
	install -m 644 $(LIB_STATIC) $(PREFIX)/lib/
	install -m 755 $(LIB_SHARED) $(PREFIX)/lib/
	install -m 644 circleDetector.h $(PREFIX)/include/

clean:
	rm -f $(LIB_OBJ) $(LIB_STATIC) $(LIB_SHARED) $(DEMO_DIR)/*.o $(DEMO_CV) $(DEMO_SPAG)

.PHONY: all demos install clean
