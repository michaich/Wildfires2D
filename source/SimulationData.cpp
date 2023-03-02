#include "SimulationData.h"
#include "Helpers.h"
#include <Cubism/HDF5Dumper.h>
#include <Cubism/HDF5Dumper_MPI.h>
#include <iomanip>
using namespace cubism;

void SimulationData::allocateGrid()
{
  ScalarLab dummy;
  const bool xper = dummy.is_xperiodic();
  const bool yper = dummy.is_yperiodic();
  const bool zper = dummy.is_zperiodic();

  const int bpdz = 1;
  T   = new ScalarGrid (bpdx,bpdy,bpdz,extent,levelStart,levelMax,comm,xper,yper,zper);
  S1  = new ScalarGrid (bpdx,bpdy,bpdz,extent,levelStart,levelMax,comm,xper,yper,zper);
  S2  = new ScalarGrid (bpdx,bpdy,bpdz,extent,levelStart,levelMax,comm,xper,yper,zper);
  tmp = new ScalarGrid (bpdx,bpdy,bpdz,extent,levelStart,levelMax,comm,xper,yper,zper);

  const std::vector<BlockInfo>& TInfo = T->getBlocksInfo();

  if (TInfo.size() == 0)
  {
    std::cout << "You are using too many MPI ranks for the given initial number of blocks.";
    std::cout << "Either increase levelStart or reduce the number of ranks." << std::endl;
    MPI_Abort(T->getWorldComm(),1);
  }
  // Compute extents, assume all blockinfos have same h at the start!!!
  const int aux = 1 << levelStart;
  extents[0] = aux * bpdx * TInfo[0].h * ScalarBlock::sizeX;
  extents[1] = aux * bpdy * TInfo[0].h * ScalarBlock::sizeY;

  // compute min and max gridspacing for set AMR parameter
  const int auxMax = 2 << (levelMax-1);
  minH = extents[0] / (auxMax*bpdx*ScalarBlock::sizeX);
  maxH = extents[0] / (bpdx*ScalarBlock::sizeX);
}

SimulationData::SimulationData() = default;

SimulationData::~SimulationData()
{
  delete profiler;
  if(T   not_eq nullptr) delete T ;
  if(S1  not_eq nullptr) delete S1;
  if(S2  not_eq nullptr) delete S2;
  if(tmp not_eq nullptr) delete tmp;
}

bool SimulationData::bOver() const
{
  return endTime>0 && time >= endTime;
}

bool SimulationData::bDump()
{
  return dumpTime>0 && time >= nextDumpTime;
}

void SimulationData::startProfiler(std::string name)
{
  profiler->push_start(name);
}

void SimulationData::stopProfiler()
{
  profiler->pop_stop();
}

void SimulationData::printResetProfiler()
{
  profiler->printSummary();
  profiler->reset();
}

double SimulationData::getH()
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

void SimulationData::dumpAll(std::string name)
{
  startProfiler("Dump");
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5_MPI<StreamerScalar,Real>(*T , time, "T_"  + ss.str(), path4serialization);
  DumpHDF5_MPI<StreamerScalar,Real>(*S1, time, "S1_" + ss.str(), path4serialization);
  DumpHDF5_MPI<StreamerScalar,Real>(*S2, time, "S2_" + ss.str(), path4serialization);
  stopProfiler();
}