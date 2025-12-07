
CXX      := clang++
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -Wpedantic -Iinclude

SRC_DIR  := src
SRCS     := $(wildcard $(SRC_DIR)/*.cpp)

TARGET   := tensorium_lang

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
