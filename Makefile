# ==========================================
# Tensorium Compiler Makefile
# ==========================================

# Compilateur et standards
CXX      := clang++
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -Wpedantic -Iinclude

# Nom de l'exécutable final
TARGET   := tensorium_cc

# Dossiers contenant les sources (.cpp)
SRC_DIRS := lib tools

# Recherche récursive de tous les fichiers .cpp dans lib/ et tools/
# Note: $(shell find ...) est standard sur Linux/macOS/WSL.
SRCS := $(shell find $(SRC_DIRS) -name "*.cpp")

# Génération de la liste des fichiers objets (.o) correspondants
OBJS := $(SRCS:.cpp=.o)

# Règle par défaut
all: $(TARGET)

# Édition de liens (Linking)
$(TARGET): $(OBJS)
	@echo "🔗 Linking executable: $@"
	$(CXX) $(CXXFLAGS) $^ -o $@
	@echo "✅ Build successful! Run with: ./$(TARGET)"

# Compilation des fichiers sources (.cpp -> .o)
%.o: %.cpp
	@echo "🔨 Compiling $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Nettoyage des fichiers générés
clean:
	@echo "🧹 Cleaning up..."
	rm -f $(OBJS) $(TARGET)

# Commandes phony (qui ne créent pas de fichiers)
.PHONY: all clean
