#pragma once

#include "SimulationData.h"
#include "Operator.h"

class Simulation
{
 public:
  SimulationData sim;
  std::vector<std::shared_ptr<Operator>> pipeline;

  Simulation(int argc, char ** argv, MPI_Comm comm);
  ~Simulation();
  void simulate();

 protected:
  cubism::ArgumentParser parser;
  double calcMaxTimestep();
  MPI_Op custom_max;
};