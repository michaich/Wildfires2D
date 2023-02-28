//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#include "Helpers.h"
#include "Cubism/HDF5Dumper_MPI.h"
#include <random>
using namespace cubism;

void IC::operator()(const double dt)
{
  const std::vector<BlockInfo>& TInfo = sim.T->getBlocksInfo();

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
      if ( std::fabs(p[0]-sim.xcenter) < 0.5*sim.xside && std::fabs(p[1]-sim.ycenter) < 0.5*sim.yside)
      {
        T(ix,iy).s = sim.Ti;
      }
      else
      {
        T(ix,iy).s = sim.Ta;
      }
    }
  }
}

double findMaxU::run() const
{
  return 1.;
  //const size_t Nblocks = velInfo.size();
  //const Real UINF = sim.uinfx, VINF = sim.uinfy;
  /////*
  //#ifdef ZERO_TOTAL_MOM
  //Real momX = 0, momY = 0, totM = 0; 
  //#pragma omp parallel for schedule(static) reduction(+ : momX, momY, totM)
  //for (size_t i=0; i < Nblocks; i++) {
  //  const Real h = velInfo[i].h;
  //  const VectorBlock& VEL = *(VectorBlock*)  velInfo[i].ptrBlock;
  //  for(int iy=0; iy<VectorBlock::sizeY; ++iy)
  //  for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
  //    const Real facMom = h*h;
  //    momX += facMom * VEL(ix,iy).u[0];
  //    momY += facMom * VEL(ix,iy).u[1];
  //    totM += facMom;
  //  }
  //}
  //Real temp[3] = {momX,momY,totM};
  //MPI_Allreduce(MPI_IN_PLACE, temp, 3, MPI_Real, MPI_SUM, sim.chi->getWorldComm());
  //momX = temp[0];
  //momY = temp[1];
  //totM = temp[2];
  ////printf("Integral of momenta X:%e Y:%e mass:%e\n", momX, momY, totM);
  //const Real DU = momX / totM, DV = momY / totM;
  //#endif
  ////*/
  //Real U = 0, V = 0, u = 0, v = 0;
  //#pragma omp parallel for schedule(static) reduction(max : U, V, u, v)
  //for (size_t i=0; i < Nblocks; i++)
  //{
  //  VectorBlock& VEL = *(VectorBlock*)  velInfo[i].ptrBlock;
  //  for(int iy=0; iy<VectorBlock::sizeY; ++iy)
  //  for(int ix=0; ix<VectorBlock::sizeX; ++ix) {
  //    #ifdef ZERO_TOTAL_MOM
  //      VEL(ix,iy).u[0] -= DU; VEL(ix,iy).u[1] -= DV;
  //    #endif
  //    U = std::max( U, std::fabs( VEL(ix,iy).u[0] + UINF ) );
  //    V = std::max( V, std::fabs( VEL(ix,iy).u[1] + VINF ) );
  //    u = std::max( u, std::fabs( VEL(ix,iy).u[0] ) );
  //    v = std::max( v, std::fabs( VEL(ix,iy).u[1] ) );
  //  }
  //}
  //Real quantities[4] = {U,V,u,v};
  //MPI_Allreduce(MPI_IN_PLACE, quantities, 4, MPI_Real, MPI_MAX, sim.chi->getWorldComm());
  //U = quantities[0];
  //V = quantities[1];
  //u = quantities[2];
  //v = quantities[3];
  //return std::max( { U, V, u, v } );
}
