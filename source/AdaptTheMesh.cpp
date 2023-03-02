#include "AdaptTheMesh.h"
#include "Helpers.h"
#include <Cubism/AMR_MeshAdaptation.h>

using namespace cubism;

//struct GradT
//{
//  GradT(const SimulationData & s) : sim(s) {}
//  const SimulationData & sim;
//  const StencilInfo stencil{-1, -1, 0, 2, 2, 1, true, {0}};
//  const std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
//  void operator()(ScalarLab & lab, const BlockInfo& info) const
//  {
//    auto& __restrict__ TMP = *(ScalarBlock*) tmpInfo[info.blockID].ptrBlock;
//    if (sim.Qcriterion)
//      for(int y=0; y<VectorBlock::sizeY; ++y)
//      for(int x=0; x<VectorBlock::sizeX; ++x)
//        TMP(x,y).s = max(TMP(x,y).s,(Real)0.0);//compress if Q<0
//
//    const int offset = (info.level == sim.tmp->getlevelMax()-1) ? 4 : 2;
//    const Real threshold = sim.bAdaptChiGradient ? 0.9 : 1e4;
//    for(int y=-offset; y<VectorBlock::sizeY+offset; ++y)
//    for(int x=-offset; x<VectorBlock::sizeX+offset; ++x)
//    {
//      lab(x,y).s = std::min(lab(x,y).s,(Real)1.0);
//      lab(x,y).s = std::max(lab(x,y).s,(Real)0.0);
//      if (lab(x,y).s > 0.0 && lab(x,y).s < threshold)
//      {
//        TMP(VectorBlock::sizeX/2-1,VectorBlock::sizeY/2  ).s = 2*sim.Rtol;
//        TMP(VectorBlock::sizeX/2-1,VectorBlock::sizeY/2-1).s = 2*sim.Rtol;
//        TMP(VectorBlock::sizeX/2  ,VectorBlock::sizeY/2  ).s = 2*sim.Rtol;
//        TMP(VectorBlock::sizeX/2  ,VectorBlock::sizeY/2-1).s = 2*sim.Rtol;
//        break;
//      }
//    }
//  }
//};


void AdaptTheMesh::operator()(const Real dt)
{
  if (sim.step % sim.AdaptSteps != 0) return;

  sim.startProfiler("AdaptTheMesh");

  const std::vector<cubism::BlockInfo>& TInfo = sim.T->getBlocksInfo();

  //GradT K(sim);
  //cubism::compute<ScalarLab>(K,sim.T);

  //We tag (mark) blocks of the grid for refinement/compression based on the magnitude of T(t,x,y)
  T_amr  ->Tag();

  //The other fields (S1,S2 and temporary) are marked according to T(t,x,y)
  S1_amr ->TagLike(TInfo);
  S2_amr ->TagLike(TInfo);
  tmp_amr->TagLike(TInfo);

  //Do the refinement/compression. Print output only for temperature (if sim.verbose = true).
  T_amr  ->Adapt(sim.time, sim.verbose,false); // last flag set to false means we interpolate values when refining
  S1_amr ->Adapt(sim.time, false      ,false); // last flag set to false means we interpolate values when refining
  S2_amr ->Adapt(sim.time, false      ,false); // last flag set to false means we interpolate values when refining
  tmp_amr->Adapt(sim.time, false      ,true ); // last flag set to true  means we do not interpolate values when refining

  sim.stopProfiler();
}