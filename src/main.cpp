
#include <iostream>

#include "../include/lexer.hpp"
#include "../include/parser.hpp"
#include "../include/printing/print_ast.hpp"
#include "../include/printing/print_indexed.hpp"
#include "../include/semantics/indexed_ast.hpp"
#include "../include/semantics/semantic_analyzer.hpp"

int main() {
  const char *src = R"(

field scalar chi
field tensor2 gamma[i,j]
field tensor2 Atilde[i,j]
field scalar alpha
metric g(t,r,theta,phi) {
    rho2  = r^2 + a^2 * cos(theta)^2
    Delta = r^2 - 2*M*r + a^2

    g(t,t)   = -(1 - 2*M/r)
    g(r,r)   = 1/(1 - 2*M/r)
}

metric g2(t,r,theta,phi) {
    rho2 = rho2   # scalar local ok
}

evolution BSSN {
    dt chi        = -2 * alpha * K
    dt gamma[i,j] = -2 * alpha * Atilde[i,j]
}
)";

  try {
    Lexer lex(src);
    Parser parser(lex);

    Program prog = parser.parseProgram();

    std::cout << "Fields:\n";
    for (const auto &f : prog.fields)
      printField(f);

    for (size_t i = 0; i < prog.metrics.size(); ++i)
      printMetric(prog.metrics[i], static_cast<int>(i));

    for (size_t i = 0; i < prog.evolutions.size(); ++i)
      printEvolution(prog.evolutions[i], static_cast<int>(i));

    SemanticAnalyzer sem(prog);

    for (const auto &m : prog.metrics) {
      IndexedMetric im = sem.analyzeMetric(m);
      printIndexedMetric(im);
    }

    for (const auto &evo : prog.evolutions) {
      IndexedEvolution ie = sem.analyzeEvolution(evo);
      printIndexedEvolution(ie);
    }

  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
