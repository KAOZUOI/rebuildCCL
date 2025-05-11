CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g -O2
LDFLAGS = -lseastar -lboost_program_options -lboost_system -lboost_filesystem -lyaml-cpp -lstdc++fs -lcares -lprotobuf -lhwloc -lnuma -lsctp -latomic -pthread

# 检测Seastar安装路径
SEASTAR_CFLAGS = $(shell pkg-config --cflags seastar 2>/dev/null || echo "-I/usr/local/include")
SEASTAR_LDFLAGS = $(shell pkg-config --libs seastar 2>/dev/null || echo "-L/usr/local/lib")

CXXFLAGS += $(SEASTAR_CFLAGS)
LDFLAGS += $(SEASTAR_LDFLAGS)

SOURCES = bootstrap_demo.cc
HEADERS = seastar_bootstrap.hpp hierarchical_bootstrap.hpp
OBJECTS = $(SOURCES:.cc=.o)
TARGET = bootstrap_demo

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

# 运行单个节点示例
run: $(TARGET)
	./$(TARGET) --rank 0 --total-ranks 4

# 运行多节点示例（在本地模拟多个节点）
run-multi: $(TARGET)
	@echo "Starting 4 nodes..."
	./$(TARGET) --rank 0 --total-ranks 4 --base-port 10000 &
	./$(TARGET) --rank 1 --total-ranks 4 --base-port 10000 &
	./$(TARGET) --rank 2 --total-ranks 4 --base-port 10000 &
	./$(TARGET) --rank 3 --total-ranks 4 --base-port 10000 &
	@echo "All nodes started. Press Ctrl+C to terminate."
	@sleep 15
	@echo "Terminating nodes..."
	@pkill -f "./$(TARGET)"

# 运行脚本，在多个终端中启动多个节点
run-terminals:
	@echo "#!/bin/bash" > run_nodes.sh
	@echo "gnome-terminal -- bash -c \"./$(TARGET) --rank 0 --total-ranks 4 --base-port 10000; read\"" >> run_nodes.sh
	@echo "gnome-terminal -- bash -c \"./$(TARGET) --rank 1 --total-ranks 4 --base-port 10000; read\"" >> run_nodes.sh
	@echo "gnome-terminal -- bash -c \"./$(TARGET) --rank 2 --total-ranks 4 --base-port 10000; read\"" >> run_nodes.sh
	@echo "gnome-terminal -- bash -c \"./$(TARGET) --rank 3 --total-ranks 4 --base-port 10000; read\"" >> run_nodes.sh
	@chmod +x run_nodes.sh
	@./run_nodes.sh
	@rm run_nodes.sh
