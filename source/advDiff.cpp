//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "advDiff.h"

using namespace cubism;

struct KernelAdvectDiffuse
{
  KernelAdvectDiffuse(const SimulationData & s) : sim(s){}
  const SimulationData & sim;
  const StencilInfo stencil{-2, -2, 0, 3, 3, 1, true, {0}};
  const std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
  const int NX = ScalarBlock::sizeX;
  const int NY = ScalarBlock::sizeY;

  void operator()(ScalarLab& T, const BlockInfo& info) const
  {
    const Real h = info.h;
    ScalarBlock & __restrict__ TMP = *(ScalarBlock*) tmpInfo[info.blockID].ptrBlock;

    for(int y=0; y<NY; ++y)
    for(int x=0; x<NX; ++x)
    {
      const double dTdx = sim.ux > 0 ? h*(2*T(x+1,y).s +3*T(x,y).s -6*T(x-1,y).s +T(x-2,y).s)/6.0 : h*(-T(x+2,y).s +6*T(x+1,y).s -3*T(x,y).s -2*T(x-1,y).s)/6.0;
      const double dTdy = sim.uy > 0 ? h*(2*T(x,y+1).s +3*T(x,y).s -6*T(x,y-1).s +T(x,y-2).s)/6.0 : h*(-T(x,y+2).s +6*T(x,y+1).s -3*T(x,y).s -2*T(x,y-1).s)/6.0;

      const double dT2dx2 = T(x+1,y).s - 2.0*T(x,y).s + T(x-1,y).s;
      const double dT2dy2 = T(x,y+1).s - 2.0*T(x,y).s + T(x,y-1).s;

      TMP(x,y).s = sim.Deffx*dT2dx2+sim.Deffy*dT2dy2-sim.ux*dTdx-sim.uy*dTdy;
    }

    BlockCase<ScalarBlock> * tempCase = (BlockCase<ScalarBlock> *)(tmpInfo[info.blockID].auxiliary);
    if (tempCase == nullptr) return;

    ScalarBlock::ElementType * faceXm = tempCase -> storedFace[0] ?  & tempCase -> m_pData[0][0] : nullptr;
    ScalarBlock::ElementType * faceXp = tempCase -> storedFace[1] ?  & tempCase -> m_pData[1][0] : nullptr;
    ScalarBlock::ElementType * faceYm = tempCase -> storedFace[2] ?  & tempCase -> m_pData[2][0] : nullptr;
    ScalarBlock::ElementType * faceYp = tempCase -> storedFace[3] ?  & tempCase -> m_pData[3][0] : nullptr;
    if (faceXm != nullptr)
    {
      const int x = 0;
      for(int y=0; y<NY; ++y) faceXm[y] = sim.Deffx*(T(x,y) - T(x-1,y));
    }
    if (faceXp != nullptr)
    {
      const int x = NX-1;
      for(int y=0; y<NY; ++y) faceXp[y] = sim.Deffx*(T(x,y) - T(x+1,y));
    }
    if (faceYm != nullptr)
    {
      const int y = 0;
      for(int x=0; x<NX; ++x) faceYm[x] = sim.Deffy*(T(x,y) - T(x,y-1));
    }
    if (faceYp != nullptr)
    {
      const int y = NY-1;
      for(int x=0; x<NX; ++x) faceYp[x] = sim.Deffy*(T(x,y) - T(x,y+1));
    }
  }
};

void advDiff::operator()(const Real dt)
{
  sim.startProfiler("advDiff");

  const std::vector<cubism::BlockInfo>&   TInfo = sim.T  ->getBlocksInfo();
  const std::vector<cubism::BlockInfo>&  S1Info = sim.S1 ->getBlocksInfo();
  const std::vector<cubism::BlockInfo>&  S2Info = sim.S2 ->getBlocksInfo();
  const std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
  const int NX = ScalarBlock::sizeX;
  const int NY = ScalarBlock::sizeY;
  const size_t Nblocks = TInfo.size();

  Trhs.resize(NX*NY*Nblocks);
  S1rhs.resize(NX*NY*Nblocks);
  S2rhs.resize(NX*NY*Nblocks);


  const double Ta = sim.initialConditions.Ta;
  const double C2 = sim.C2;
  const double C3 = sim.C3;
  const double C4 = sim.C4;

  KernelAdvectDiffuse K(sim) ;
  cubism::compute<ScalarLab>(K,sim.T,sim.tmp);
  #pragma omp parallel for
  for (size_t i=0; i < Nblocks; i++)
  {
    ScalarBlock & __restrict__ T   = *(ScalarBlock*)   TInfo[i].ptrBlock;
    ScalarBlock & __restrict__ S1  = *(ScalarBlock*)  S1Info[i].ptrBlock;
    ScalarBlock & __restrict__ S2  = *(ScalarBlock*)  S2Info[i].ptrBlock;
    ScalarBlock & __restrict__ TMP = *(ScalarBlock*) tmpInfo[i].ptrBlock;
    const double ih2 = 1.0/(TInfo[i].h*TInfo[i].h);
    for(int iy=0; iy<NY; ++iy)
    for(int ix=0; ix<NX; ++ix)
    {
      const int idx = i*NX*NY + iy*NX + ix;
      const double AdvectionDiffusion = ih2 * TMP(ix,iy).s;
      const double S = S1(ix,iy).s + S2(ix,iy).s;
      const double C0inv = 1.0/(sim.a*S + (1.0-sim.a)*sim.lambda*sim.gamma + sim.a*sim.gamma*(1.0-S));
      const double C1 = 1/C0inv - sim.a*S;
      const double r1 = sim.cs1 * exp(-sim.b1/T(ix,iy).s);
      const double r2 = sim.cs2 * exp(-sim.b2/T(ix,iy).s);
      const double r2t = sim.rm*r2/(sim.rm+r2);
      const double U = sim.Anc*pow(T(ix,iy).s-Ta,1.0/3.0) + sim.epsilon*sim.sigmab*(T(ix,iy).s*T(ix,iy).s+Ta*Ta)*(T(ix,iy).s+Ta);
      Trhs[idx] = (C1*AdvectionDiffusion - C2*S1(ix,iy).s*r1 + C3*S2(ix,iy).s*r2t - C4*U*(T(ix,iy).s-Ta) )*C0inv;
      S1rhs[idx] = -S1(ix,iy).s*r1;
      S2rhs[idx] = -S2(ix,iy).s*r2t;
    }
  }

  #pragma omp parallel for
  for (size_t i=0; i < Nblocks; i++)
  {
    ScalarBlock & __restrict__ T   = *(ScalarBlock*)   TInfo[i].ptrBlock;
    ScalarBlock & __restrict__ S1  = *(ScalarBlock*)  S1Info[i].ptrBlock;
    ScalarBlock & __restrict__ S2  = *(ScalarBlock*)  S2Info[i].ptrBlock;
    for(int iy=0; iy<NY; ++iy)
    for(int ix=0; ix<NX; ++ix)
    {
      const int idx = i*NX*NY + iy*NX + ix;
      T (ix,iy).s += dt* Trhs[idx];
      S1(ix,iy).s += dt*S1rhs[idx];
      S2(ix,iy).s += dt*S2rhs[idx];
    }
  }

  sim.stopProfiler();
}
