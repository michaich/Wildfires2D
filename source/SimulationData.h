#pragma once

#include "Definitions.h"
#include "Cubism/Profiler.h"
#include <memory>


struct SimulationData
{
  // MPI
  MPI_Comm comm;
  int rank;

  /* parsed parameters */
  /*********************/

  // blocks per dimension
  int bpdx;
  int bpdy;

  // number of levels
  int levelMax;

  // initial level
  int levelStart;

  // refinement/compression tolerance for voriticy magnitude
  double Rtol;
  double Ctol;

  //check for mesh refinement every this many steps
  int AdaptSteps;

  // maximal simulation extent (direction with max(bpd))
  double extent;

  // simulation extents
  std::array<double,2> extents;

  // timestep / cfl condition
  double dt;
  double CFL;

  // simulation ending parameters
  int nsteps;
  double endTime;

  // output setting
  double dumpTime;
  bool verbose;
  std::string path4serialization;
  std::string path2file;

  // initialize profiler
  cubism::Profiler * profiler = new cubism::Profiler();

  // scalar fields
  ScalarGrid * T  = nullptr; //temperature
  ScalarGrid * S1 = nullptr; //endothermic fuel mass fraction
  ScalarGrid * S2 = nullptr; //exothermic fuel mass fraction

  // parameters/constants of the model
  double a;
  double cs1;
  double cs2;
  double b1;
  double b2;
  double rm;
  double Anc;
  double epsilon;
  double sigmab;
  double H;
  double rhos;
  double rhog;
  double cps;
  double cpg;
  double A1;
  double A2;
  double Dbuoyx;
  double Dbuoyy;
  double Ad;

  // initial condition parameters
  double Ta;
  double Ti;
  double xcenter;
  double ycenter;
  double xside;
  double yside;

  // simulation time
  double time = 0;

  // simulation step
  int step = 0;

  // time of next dump
  double nextDumpTime = 0;

  // bools specifying whether we dump or not
  bool _bDump = false;

  void allocateGrid();
  bool bDump();
  void registerDump();
  bool bOver() const;

  // minimal and maximal gridspacing possible
  double minH;
  double maxH;

  SimulationData();
  SimulationData(const SimulationData &) = delete;
  SimulationData(SimulationData &&) = delete;
  SimulationData& operator=(const SimulationData &) = delete;
  SimulationData& operator=(SimulationData &&) = delete;
  ~SimulationData();

  // minimal gridspacing present on grid
  double getH()
  {
    double minHGrid = std::numeric_limits<double>::infinity();
    auto & infos = T->getBlocksInfo();
    for (size_t i = 0 ; i< infos.size(); i++)
    {
      minHGrid = std::min((double)infos[i].h, minHGrid);
    }
    MPI_Allreduce(MPI_IN_PLACE, &minHGrid, 1, MPI_Real, MPI_MIN, comm);
    return minHGrid;
  }

  void dumpAll(std::string name);
  void startProfiler(std::string name);
  void stopProfiler();
  void printResetProfiler();
};