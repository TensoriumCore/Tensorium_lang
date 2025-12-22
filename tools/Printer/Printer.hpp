#pragma once

#include "tensorium/AST/AST.hpp"
#include "tensorium/AST/IndexedAST.hpp"
#include <iostream>

namespace tensorium {

void printField(const FieldDecl &f);

void printSimulation(const SimulationConfig &sim);

void printMetric(const MetricDecl &m, int idx);
void printIndexedMetric(const IndexedMetric &m);

void printEvolution(const EvolutionDecl &evo, int idx);
void printIndexedEvolution(const IndexedEvolution &evo);

void printIndexedExpr(const IndexedExpr *e);

} // namespace tensorium
