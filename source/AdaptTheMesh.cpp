#include "AdaptTheMesh.h"
#include "Helpers.h"
#include <Cubism/AMR_MeshAdaptation.h>

using namespace cubism;

//compute grad(T) and mark field 'tmp' based on grad(T) and on T so that mesh is refined/compressed correctly 
struct GradT
{
  GradT(const SimulationData & s) : sim(s) {}
  const SimulationData & sim;
  const StencilInfo stencil{-1, -1, 0, 2, 2, 1, false, {0}};
  const std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
  void operator()(ScalarLab & lab, const BlockInfo& info) const
  {
    auto& __restrict__ TMP = *(ScalarBlock*) tmpInfo[info.blockID].ptrBlock;
    const double aux = 0.5/info.h;

    for(int y=0; y<ScalarBlock::sizeY; ++y)
    for(int x=0; x<ScalarBlock::sizeX; ++x)
    {
      const double dTdx = aux*(lab(x+1,y).s-lab(x-1,y).s);
      const double dTdy = aux*(lab(x,y+1).s-lab(x,y-1).s);
      if (std::fabs(dTdx) > sim.gradRtol || std::fabs(dTdy) > sim.gradRtol || lab(x,y).s > sim.TRtol)
      {
        TMP(x,y).s = 10.0; //large value, refine the mesh here
      }
      else if (std::fabs(dTdx) < sim.gradCtol && std::fabs(dTdy) < sim.gradCtol && lab(x,y).s < sim.TCtol)
      {
        TMP(x,y).s = 0.0; //small value, compress the mesh here        
      }
      else
      {
        TMP(x,y).s = 0.1; //do nothing        
      }
    }
  }
};


void AdaptTheMesh::operator()(const Real dt)
{
  if (sim.step % sim.AdaptSteps != 0) return;

  sim.startProfiler("AdaptTheMesh");

  const std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();

  GradT K(sim);
  cubism::compute<ScalarLab>(K,sim.T);

  //We tag (mark) blocks of the grid for refinement/compression based on the magnitude of T(t,x,y) and of grad T
  tmp_amr  ->Tag();

  //The other fields (S1,S2 and temporary) are marked according to T(t,x,y)
  T_amr  ->TagLike(tmpInfo);
  S1_amr ->TagLike(tmpInfo);
  S2_amr ->TagLike(tmpInfo);
  S2_0_amr ->TagLike(tmpInfo);

  //Do the refinement/compression. Print output only for temperature (if sim.verbose = true).
  T_amr  ->Adapt(sim.time, sim.verbose,false); // last flag set to false means we interpolate values when refining
  S1_amr ->Adapt(sim.time, false      ,false); // last flag set to false means we interpolate values when refining
  S2_amr ->Adapt(sim.time, false      ,false); // last flag set to false means we interpolate values when refining
  S2_0_amr ->Adapt(sim.time, false      ,false); // last flag set to false means we interpolate values when refining
  tmp_amr->Adapt(sim.time, false      ,true ); // last flag set to true  means we do not interpolate values when refining

  sim.stopProfiler();
}