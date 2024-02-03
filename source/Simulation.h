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

  //Quantity of interest, average temperature in rectangle [xA,xB]x[yA,yB]
  double xA,yA,xB,yB;
  std::vector<double> Taverage;
  std::vector<double> Vaverage;
  std::vector<double> Tsave;

 protected:
  cubism::ArgumentParser parser;
  double calcMaxTimestep();
};