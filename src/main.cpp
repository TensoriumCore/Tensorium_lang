#include "../include/lexer.hpp"
#include "../include/parser.hpp"
#include "../include/printing/print_ast.hpp"
#include "../include/printing/print_indexed.hpp"
#include "../include/semantics/semantic_analyzer.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

struct TestCase {
  std::string name;
  std::string input;
  bool expectFailure = false;
};

bool runTest(const TestCase &t) {
  try {
    Lexer lex(t.input.c_str());
    Parser parser(lex);
    Program prog = parser.parseProgram();
    SemanticAnalyzer sem(prog);

    std::vector<IndexedMetric> indexedMetrics;
    std::vector<IndexedEvolution> indexedEvos;

    for (auto &m : prog.metrics)
      indexedMetrics.push_back(sem.analyzeMetric(m));

    for (auto &e : prog.evolutions)
      indexedEvos.push_back(sem.analyzeEvolution(e));

    if (t.expectFailure) {
      std::cerr << " FAIL (unexpected success): " << t.name << "\n";
      return false;
    }

    std::cout << " PASS: " << t.name << "\n\n";

    std::cout << "========== Pretty-print ==========\n";
    std::cout << "Fields:\n";
    for (auto &f : prog.fields)
      printField(f);

    for (size_t i = 0; i < prog.metrics.size(); i++) {
      printMetric(prog.metrics[i], (int)i);
      printIndexedMetric(indexedMetrics[i]);
    }

    for (size_t i = 0; i < prog.evolutions.size(); i++) {
      printEvolution(prog.evolutions[i], (int)i);
      printIndexedEvolution(indexedEvos[i]);
    }
    std::cout << "==================================\n\n";

    return true;

  } catch (const std::exception &ex) {
    if (t.expectFailure) {
      std::cout << "✔ PASS (expected failure): " << t.name << " -- "
                << ex.what() << "\n";
      return true;
    }
    std::cerr << " FAIL: " << t.name << " -- " << ex.what() << "\n";
    return false;
  }
}

int main() {

  std::vector<TestCase> tests = {

      {"Valid BSSN evolution",
       R"(
            field scalar chi
            field cov_tensor2 gamma[i,j]
            field cov_tensor2 Atilde[i,j]
            field scalar alpha

            metric g(t,r,theta,phi) {
                rho2 = r^2 + a^2 * cos(theta)^2
                g(t,t) = -(1 - 2*M/r)
            }

            evolution BSSN {
                dt chi        = -2 * alpha * K
                dt gamma[i,j] = -2 * alpha * Atilde[i,j]
                Atilde[i,j]   = contract(gamma[i,k] * Atilde[k,j])
            }
        )"},

      {"Scalar evolution OK",
       R"(
            field scalar phi

            evolution Test {
                dt phi = 2 * phi
            }
        )"},

      {
		"Correct nested contraction",
        R"(
		    field cov_tensor2 A[i,j]
	
		    evolution OK {
				dt A[i,j] = contract(A[i,k] * (A[k,l] * A[l,j]))
			}
		)",
          true 
      },

      {"Local temporary reuse OK",
       R"(
            field scalar chi

            evolution Temp {
                dt chi = K
                K = chi * chi
            }
        )"},

      {"Metric-only parameters allowed",
       R"(
            field scalar rho

            metric g(t,r) {
                test = r + t
            }

            evolution OK {
                dt rho = r
            }
        )"},

      {"Invalid index",
       R"(
            field cov_tensor2 gamma[i,j]

            evolution Wrong {
                dt gamma[i,j] = gamma[i,p]  # p not allowed
            }
        )",
       true},

      {"Missing contraction",
       R"(
            field cov_tensor2 gamma[i,j]

            evolution Wrong {
                dt gamma[i,j] = gamma[i,k]  # k appears only on RHS
            }
        )",
       true},

      {"Too many repeated indices (illegal contraction)",
       R"(
            field cov_tensor2 A[i,j]

            evolution Bad {
                dt A[i,j] = A[i,k] * A[k,k] * A[k,j]
            }
        )",
       true},

      {"Wrong tensor rank - vector indexed like matrix",
       R"(
            field vector beta[i]

            evolution Bad {
                dt beta[i,j] = beta[i]
            }
        )",
       true},

      {"Local variable shadowing error",
       R"(
            field scalar chi

            evolution Bad {
                dt chi = alpha
                chi = 5
            }
        )",
       true},

      {"Function call misused in tensor context",
       R"(
            field cov_tensor2 gamma[i,j]
            field cov_tensor2 Atilde[i,j]

            evolution Bad {
                dt gamma[i,j] = sin(gamma[i,j])
            }
        )",
       true},

      {"Reference to undeclared field",
       R"(
            field scalar phi

            evolution Bad {
                dt psi = phi
            }
        )",
       true},

      {"Index missing on RHS",
       R"(
            field cov_tensor2 T[i,j]

            evolution Bad {
                dt T[i,j] = T[i,k]
            }
        )",
       true},
	{
    "Partial derivative on scalar -> covector",
    R"(
        field scalar phi
        field covector grad_phi[i]

        evolution Diff {
            dt grad_phi[i] = d_i(phi)
        }
    )",
	},
	{
    "Laplacian expects scalar",
    R"(
        field covector v[i]

        evolution BadLap {
            dt v[i] = laplacian(v[i])
        }
    )",
    true
},
  };

  bool ok = true;
  for (auto &t : tests)
    ok &= runTest(t);

  std::cout << "\n=== FINAL TEST STATUS: " << (ok ? "ALL PASSED ✔" : "FAIL ")
            << "\n";

  return ok ? 0 : 1;
}
