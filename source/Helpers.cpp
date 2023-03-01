#include "Helpers.h"
#include <random>
using namespace cubism;

void IC::operator()(const double dt)
{
  const std::vector<BlockInfo>& TInfo = sim.T->getBlocksInfo();

  const double Ti = sim.initialConditions.Ti;
  const double Ta = sim.initialConditions.Ta;
  const double xcenter = sim.initialConditions.xcenter;
  const double ycenter = sim.initialConditions.ycenter;
  const double xside = sim.initialConditions.xside;
  const double yside = sim.initialConditions.yside;

  #pragma omp parallel for
  for (size_t i=0; i < TInfo.size(); i++)
  {
    auto & T  = (*sim.T)(i);
    auto & S1 = (*sim.S1)(i);
    auto & S2 = (*sim.S2)(i);
    for(int iy=0; iy<ScalarBlock::sizeY; ++iy)
    for(int ix=0; ix<ScalarBlock::sizeX; ++ix)
    {
      double p[2];
      TInfo[i].pos(p,ix,iy);
      S1(ix,iy).s = .15;
      S2(ix,iy).s = .85;
      if ( std::fabs(p[0]-xcenter) < 0.5*xside && std::fabs(p[1]-ycenter) < 0.5*yside)
      {
        T(ix,iy).s = Ti;
      }
      else
      {
        T(ix,iy).s = Ta;
      }
    }
  }
}