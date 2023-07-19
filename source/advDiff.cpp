#include "advDiff.h"

using namespace cubism;

struct KernelAdvectDiffuse
{
  KernelAdvectDiffuse(const SimulationData & s) : sim(s){}
  const SimulationData & sim;
  const StencilInfo stencil{-2, -2, 0, 3, 3, 1, false, {0}};
  const std::vector<cubism::BlockInfo>& tmpInfo = sim.tmp->getBlocksInfo();
  const int NX = ScalarBlock::sizeX;
  const int NY = ScalarBlock::sizeY;
  const double c[4] = {2.0/6.0,3.0/6.0,-6.0/6.0,1.0/6.0}; //stencil coefficients for 3rd-order upwind first derivative

  void operator()(ScalarLab& T, const BlockInfo& info) const
  {
    const Real h = info.h;
    ScalarBlock & __restrict__ TMP = *(ScalarBlock*) tmpInfo[info.blockID].ptrBlock;

    for(int y=0; y<NY; ++y)
    for(int x=0; x<NX; ++x)
    {
      double ux = 0.0;
      double uy = 0.0;
      sim.getVelocity(info.blockID,x,y,ux,uy);

      const double Deffx = sim.Drb + sim.Ad * ux * sim.Lcx;
      const double Deffy = sim.Drb + sim.Ad * uy * sim.Lcy;

      const double dTdx = ux > 0 ? h*(c[0]*T(x+1,y).s + c[1]*T(x,y).s + c[2]*T(x-1,y).s + c[3]*T(x-2,y).s) :
                                  -h*(c[0]*T(x-1,y).s + c[1]*T(x,y).s + c[2]*T(x+1,y).s + c[3]*T(x+2,y).s);
      const double dTdy = uy > 0 ? h*(c[0]*T(x,y+1).s + c[1]*T(x,y).s + c[2]*T(x,y-1).s + c[3]*T(x,y-2).s) :
                                  -h*(c[0]*T(x,y-1).s + c[1]*T(x,y).s + c[2]*T(x,y+1).s + c[3]*T(x,y+2).s);
      const double dT2dx2 = T(x+1,y).s - 2.0*T(x,y).s + T(x-1,y).s;
      const double dT2dy2 = T(x,y+1).s - 2.0*T(x,y).s + T(x,y-1).s;
      TMP(x,y).s = Deffx*dT2dx2+Deffy*dT2dy2-ux*dTdx-uy*dTdy;
    }

    BlockCase<ScalarBlock> * tempCase = (BlockCase<ScalarBlock> *)(tmpInfo[info.blockID].auxiliary);
    if (tempCase == nullptr) return;

    ScalarBlock::ElementType * faceXm = tempCase -> storedFace[0] ?  & tempCase -> m_pData[0][0] : nullptr;
    ScalarBlock::ElementType * faceXp = tempCase -> storedFace[1] ?  & tempCase -> m_pData[1][0] : nullptr;
    ScalarBlock::ElementType * faceYm = tempCase -> storedFace[2] ?  & tempCase -> m_pData[2][0] : nullptr;
    ScalarBlock::ElementType * faceYp = tempCase -> storedFace[3] ?  & tempCase -> m_pData[3][0] : nullptr;
    double ux = 0.0;
    double uy = 0.0;
    if (faceXm != nullptr)
    {
      const int x = 0;
      for(int y=0; y<NY; ++y)
      {
        sim.getVelocity(info.blockID,x,y,ux,uy);
        const double Deffx = sim.Drb + sim.Ad * ux * sim.Lcx;
        faceXm[y] = Deffx*(T(x,y) - T(x-1,y));
        faceXm[y] -= ux*h/6.0* (ux > 0 ? (2.0*T(x  ,y)+5.0*T(x-1,y)-T(x-2,y)):(-2.0*T(x  ,y)-5.0*T(x+1,y)+T(x+2,y)) );
      }
    }
    if (faceXp != nullptr)
    {
      const int x = NX-1;
      for(int y=0; y<NY; ++y)
      {
        sim.getVelocity(info.blockID,x,y,ux,uy);
        const double Deffx = sim.Drb + sim.Ad * ux * sim.Lcx;
        faceXp[y] = Deffx*(T(x,y) - T(x+1,y));
        faceXp[y] += ux*h/6.0* (ux > 0 ? (2.0*T(x+1,y)+5.0*T(x  ,y)-T(x-1,y)):(-2.0*T(x-1,y)-5.0*T(x  ,y)+T(x+1,y)) );
      }
    }
    if (faceYm != nullptr)
    {
      const int y = 0;
      for(int x=0; x<NX; ++x)
      {
        sim.getVelocity(info.blockID,x,y,ux,uy);
        const double Deffy = sim.Drb + sim.Ad * uy * sim.Lcy;
        faceYm[x] = Deffy*(T(x,y) - T(x,y-1));
        faceYm[x] -= uy*h/6.0* (uy > 0 ? (2.0*T(x,y  )+5.0*T(x,y-1)-T(x,y-2)):(-2.0*T(x,y  )-5.0*T(x,y+1)+T(x,y+2)) );
      }
    }
    if (faceYp != nullptr)
    {
      const int y = NY-1;
      for(int x=0; x<NX; ++x)
      {
        sim.getVelocity(info.blockID,x,y,ux,uy);
        const double Deffy = sim.Drb + sim.Ad * uy * sim.Lcy;
        faceYp[x] = Deffy*(T(x,y) - T(x,y+1));
        faceYp[x] += uy*h/6.0* (uy > 0 ? (2.0*T(x,y+1)+5.0*T(x,y  )-T(x,y-1)):(-2.0*T(x,y-1)-5.0*T(x,y  )+T(x,y+1)) );
      }
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

  Trhs .resize(NX*NY*Nblocks);
  S1rhs.resize(NX*NY*Nblocks);
  S2rhs.resize(NX*NY*Nblocks);

  const double Ta      = sim.initialConditions.Ta;
  const double C2      = sim.C2;
  const double C3      = sim.C3;
  const double C4      = sim.C4;
  const double b1      = sim.b1;
  const double b2      = sim.b2;
  const double rm0     = sim.rm0;
  const double rmc     = sim.rmc;
  const double a       = sim.a;
  const double cs1     = sim.cs1;
  const double cs2     = sim.cs2;
  const double Anc     = sim.Anc;
  const double epsilon = sim.epsilon;
  const double sigmab  = sim.sigmab;
  const double gamma   = sim.gamma;
  const double lambda  = sim.lambda;

  //Low-storage 3rd-order Runge Kutta time integration

  //RK3 coefficients
  const Real alpha[3] = { 1.0/3.0,  15.0/ 16.0,8.0/15.0};
  const Real  beta[3] = {-5.0/9.0,-153.0/128.0,0.0     };

  //We need a right-hand-side (RHS) vector for each field, to perform RK3
  std::fill(Trhs .begin(), Trhs .end(), 0.0);
  std::fill(S1rhs.begin(), S1rhs.end(), 0.0);
  std::fill(S2rhs.begin(), S2rhs.end(), 0.0);

  //This struct will compute the advection and diffusion terms that use a stencil for the derivatives
  KernelAdvectDiffuse K(sim);

  //Perform the three RK steps
  for (int RKstep = 0; RKstep < 3; RKstep ++)
  {
    //Compute advection+diffusion terms and store result to grid 'tmp'
    cubism::compute<ScalarLab>(K,sim.T,sim.tmp);

    //Loop over all gridpoints and compute the RHS of the PDEs we are solving
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
        double ux;
        double uy;
        sim.getVelocity(i,ix,iy,ux,uy);
        const double ut = std::sqrt(ux*ux+uy*uy);

        //match grid point 'idx' with grid point (ix,iy) from block i
        const int idx = i*NX*NY + iy*NX + ix;

        //advection+diffusion terms computed from before
        const double AdvectionDiffusion = ih2 * TMP(ix,iy).s;

        //auxiliary quantities needed for right-hand sides
        const double S     = S1(ix,iy).s + S2(ix,iy).s;
        const double C0inv = 1.0/(a*S + (1.0-a)*lambda*gamma + a*gamma*(1.0-S));
        const double C1    = 1.0/C0inv - a*S;
        const double r1    = cs1 * exp(-b1/T(ix,iy).s);
        const double r2    = cs2 * exp(-b2/T(ix,iy).s);
        const double rm    = rm0 + rmc * std::max(ut-1.0,0.0);
        const double r2t   = rm*r2/(rm+r2);
        const double C4U   = C4*(Anc*pow(T(ix,iy).s-Ta,1.0/3.0) + epsilon*sigmab*(T(ix,iy).s*T(ix,iy).s+Ta*Ta)*(T(ix,iy).s+Ta));

        //right-hand sides
        Trhs [idx] += dt*(C1*AdvectionDiffusion - C2*r1*S1(ix,iy).s + C3*r2t*S2(ix,iy).s - C4U*(T(ix,iy).s-Ta))*C0inv;
        S1rhs[idx] += -dt*S1(ix,iy).s*r1;
        S2rhs[idx] += -dt*S2(ix,iy).s*r2t;

        //perform RK step here
        T (ix,iy).s += Trhs [idx]*alpha[RKstep];
        S1(ix,iy).s += S1rhs[idx]*alpha[RKstep];
        S2(ix,iy).s += S2rhs[idx]*alpha[RKstep];
        Trhs [idx] *= beta[RKstep];
        S1rhs[idx] *= beta[RKstep];
        S2rhs[idx] *= beta[RKstep];
      }
    }
  }

  sim.stopProfiler();
}
