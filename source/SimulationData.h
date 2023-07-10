#pragma once
using Real = double;
#define MPI_Real MPI_DOUBLE

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <omp.h>
#include <memory>
#include <Cubism/ArgumentParser.h>
#include <Cubism/Grid.h>
#include <Cubism/GridMPI.h>
#include <Cubism/BlockInfo.h>
#include <Cubism/BlockLab.h>
#include <Cubism/BlockLabMPI.h>
#include <Cubism/StencilInfo.h>
#include <Cubism/AMR_MeshAdaptation.h>
#include <Cubism/Definitions.h>
#include "Cubism/Profiler.h"

using ScalarElement = cubism::ScalarElement<double>;
using ScalarBlock   = cubism::GridBlock<_BS_,2,ScalarElement>;
using ScalarGrid    = cubism::GridMPI<cubism::Grid<ScalarBlock, std::allocator>>;
using ScalarLab     = cubism::BlockLabMPI<cubism::BlockLabNeumann  <ScalarGrid, 2,std::allocator>>;
using ScalarAMR     = cubism::MeshAdaptation<ScalarLab>;


struct SimulationData
{
  // MPI parameters
  MPI_Comm comm;
  int rank;

  // Adaptive Mesh Refinement parameters
  int bpdx;       // blocks in x-direction at refinement level 0
  int bpdy;       // blocks in y-direction at refinement level 0
  int levelMax;   // maximum number of refinement levels
  int levelStart; // the mesh starts (at t=0) as a uniform mesh at refinement level 'levelStart'
  double TRtol;    // refine the mesh if T(t,x,y) > TRtol
  double TCtol;    // compress the mesh if T(t,x,y) < TCtol
  double gradRtol;    // refine the mesh if grad T(t,x,y) > gradRtol
  double gradCtol;    // compress the mesh if grad T(t,x,y) < gradCtol
  int AdaptSteps; // check for mesh refinement/compression once every 'AdaptSteps' timesteps
  double minH;    // minimum grid spacing possible (at level = levelmax - 1)
  double maxH;    // maximum grid spacing possible (at level = 0)

  std::array<double,2> extents; // simulation is in rectangle [0,extents[0]] x [0,extents[1]]
  double extent; // equal to extents[0] if bpdx > bpdy, otherwise equal to extents[1]. 
  //User provides 'bpdx', 'bpdy' and 'extent'.
  //'extents' are then determined from those three numbers.

  // time stepping parameters
  double dt;       // current timestep size
  double CFL;      // CFL condition, to determine max possible timestep
  double endTime;  // simulation ends when t=endTime
  double time = 0; // current time
  int step = 0;    // current timestep number

  // output settings
  double dumpTime;                                      // save the fields every 'dumpTime' time
  bool verbose;                                         // print more screen outpout if true
  std::string path4serialization;                       // path where all results are stored
  cubism::Profiler * profiler = new cubism::Profiler(); // profiler to measure execution times
  double nextDumpTime = 0;                              // time of next save

  // scalar fields
  ScalarGrid * T  = nullptr; //temperature
  ScalarGrid * S1 = nullptr; //endothermic fuel mass fraction
  ScalarGrid * S2 = nullptr; //exothermic fuel mass fraction
  ScalarGrid * tmp = nullptr; //auxiliary temporary grid

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
  struct InitialConditions
  {
    double Ta; //temperature without fire (~300K)

    //ignition zones
    int number_of_zones;
    std::vector<double> Ti;
    std::vector<double> xignition;
    std::vector<double> yignition;
    std::vector<double> xside_ignition;
    std::vector<double> yside_ignition;

    //roads (zero starting fuel)
    int number_of_roads;
    std::vector<double> Troad;
    std::vector<double> xroad;
    std::vector<double> yroad;
    std::vector<double> xside_road;
    std::vector<double> yside_road;
  };
  InitialConditions initialConditions;

  // velocity field parameters
  double delta;
  double eta;
  double z0;
  double kappa;
  double u10x;
  double u10y;


  double ux; // x-component of velocity field (constant, at least for now)
  double uy; // y-component of velocity field (constant, at least for now)

  // constant terms that depend of the other parameters of the model
  double C2;
  double C3;
  double C4;
  double gamma;
  double lambda;

  // dispersion coefficients
  double Deffx;
  double Deffy;

  // characteristic fire lengths
  double Lcx;
  double Lcy;

  // endothermic and exothermic initial condition bounds
  int ic_seed;
  double S1min,S1max;
  double S2min,S2max;

  void allocateGrid();                  // called when simulation starts, to allocate the gridds
  bool bDump();                         // check if dumping of fields is needed at the current time
  bool bOver() const;                   // check if simulation should terminate
  double getH();                        // find smallest grid spacing currently present on the grid
  void dumpAll(std::string name);       // save files under name 'name'
  void startProfiler(std::string name); // start measuring execution time of 'name'
  void stopProfiler();                  // stop measuring execution time
  void printResetProfiler();            // print measured execution times

  SimulationData();  //class contructor
  ~SimulationData(); //class destructor

  //don't allow for copy creation of this class (to avoid bugs)
  SimulationData(const SimulationData &) = delete;
  SimulationData(SimulationData &&) = delete;
  SimulationData& operator=(const SimulationData &) = delete;
  SimulationData& operator=(SimulationData &&) = delete;
};
