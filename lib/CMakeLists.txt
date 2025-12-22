add_library(tensorium)

target_sources(tensorium
  PRIVATE
    AST/ASTPrinter.cpp
    AST/Expr.cpp
    Lex/Lexer.cpp
    Parse/Parser.cpp
    Sema/Sema.cpp
    Backend/BackendBuilder.cpp
	Runtime/Eval.cpp
	Runtime/CpuRuntime.cpp
)

target_include_directories(tensorium
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(tensorium PUBLIC cxx_std_20)
