ifneq ($(C_ENV), dev)
	CC=nvcc
	CFLAGS=-O3 -gencode arch=compute_30,code=sm_30
else
	CC=/usr/local/cuda-6.5/bin/nvcc
	CFLAGS=-O3 -gencode arch=compute_30,code=sm_30
endif

LDFLAGS=-lczmq -lzmq
SOURCE=./src
TARGET=./build

all: dir $(TARGET)/ventilator $(TARGET)/worker $(TARGET)/sink $(TARGET)/ntt $(TARGET)/sequential

dir:
	mkdir -p $(TARGET)

$(TARGET)/ventilator: $(SOURCE)/taskven.cu
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(TARGET)/ventilator $(SOURCE)/taskven.cu

$(TARGET)/worker: $(SOURCE)/taskworker.cu
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(TARGET)/worker $(SOURCE)/taskworker.cu

$(TARGET)/sink: $(SOURCE)/tasksink.cu
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(TARGET)/sink $(SOURCE)/tasksink.cu

$(TARGET)/ntt: $(SOURCE)/nfft.cu
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(TARGET)/ntt $(SOURCE)/nfft.cu

$(TARGET)/sequential: $(SOURCE)/sequential.cu
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(TARGET)/sequential $(SOURCE)/sequential.cu

clean:
	rm -rf $(TARGET)/*
