CXX      := clang++
CXXFLAGS := -std=c++20 -O3 -mtune=native -Wall -Wextra -Wpedantic -Iinclude

TARGET   := Tensorium_cc

SRC_DIRS := lib tools tools/Printer

SRCS := $(shell find $(SRC_DIRS) -name "*.cpp")

OBJS := $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Linking executable: $@"
	$(CXX) $(CXXFLAGS) $^ -o $@
	@echo "Build successful! Run with: ./$(TARGET)"

%.o: %.cpp
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@echo "Cleaning up..."
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
