CC=g++
CFLAGS=-std=c++11 -O3 -Wall
LDFLAGS=-lzmqpp -lzmq
SOURCE=./src
TARGET=./build

all: dir $(TARGET)/ventilator $(TARGET)/worker $(TARGET)/sink

dir:
	mkdir -p $(TARGET)

$(TARGET)/ventilator: $(SOURCE)/taskven.cc
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(TARGET)/ventilator $(SOURCE)/taskven.cc


$(TARGET)/worker: $(SOURCE)/taskworker.cc
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(TARGET)/worker $(SOURCE)/taskworker.cc


$(TARGET)/sink: $(SOURCE)/tasksink.cc
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(TARGET)/sink $(SOURCE)/tasksink.cc


clean: 
	rm -rf $(TARGET)/*
