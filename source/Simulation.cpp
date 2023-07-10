#include "Simulation.h"
#include <Cubism/HDF5Dumper.h>
#include "Helpers.h"
#include "advDiff.h"
#include "AdaptTheMesh.h"
#include <algorithm>
#include <iterator>

using namespace cubism;

Simulation::Simulation(int argc, char ** argv, MPI_Comm comm) : parser(argc,argv)
{
  //1. Print initial general information
  {
    sim.comm = comm;
    MPI_Comm_rank(sim.comm,&sim.rank);
    if (sim.rank == 0)
    {
      int size;
      MPI_Comm_size(sim.comm,&size);
      std::cout <<"==========================\n";
      std::cout <<"    Wildfire spread 2D    \n";
      std::cout <<"==========================\n";
      parser.print_args();
      #pragma omp parallel
      {
        #pragma omp master
        std::cout << "[WS2D] Running with " << size << " rank(s) and " << omp_get_num_threads() << " thread(s).\n"; 
      }
    }
  }

  //2. Read input parameters
  {
    if (sim.verbose) std::cout << "[WS2D] Parsing Simulation Configuration..." << std::endl;

    //2a. Parameters that have to be given.
    parser.set_strict_mode();
  
    //2b. Parameters that can be given. If not given, default values are used.
    parser.unset_strict_mode();

    sim.bpdx     = parser("-bpdx"    ).asInt(2)   ; //initial number of blocks in x-direction
    sim.bpdy     = parser("-bpdy"    ).asInt(2)   ; //initial number of blocks in y-direction
    sim.levelMax = parser("-levelMax").asInt(6)   ; //max number of refinement levels

    sim.TRtol     = parser("-TRtol"    ).asDouble(800); //tolerance for mesh refinement
    sim.TCtol     = parser("-TCtol"    ).asDouble(400); //tolerance for mesh compression
    sim.gradRtol  = parser("-gradRtol" ).asDouble(10 ); //tolerance for mesh refinement (grad T)
    sim.gradCtol  = parser("-gradCtol" ).asDouble(0.1); //tolerance for mesh compression (grad T)



    sim.AdaptSteps = parser("-AdaptSteps").asInt(20)              ; //check for refinement every this many timesteps
    sim.levelStart = parser("-levelStart").asInt(sim.levelMax - 1); //initial level of refinement
    sim.extent     = parser("-extent"    ).asDouble(1000)         ; //simulation extent
    sim.CFL        = parser("-CFL"       ).asDouble(0.5)          ; //CFL number
    sim.endTime    = parser("-tend"      ).asDouble(10000. )      ; //simulation ends at t=tend (inactive if tend=0)
  
    // output parameters
    sim.dumpTime           = parser("-tdump"        ).asDouble(10.);
    sim.path4serialization = parser("-serialization").asString("./");
    sim.verbose            = parser("-verbose"      ).asInt(1);
    sim.verbose = sim.verbose && sim.rank == 0;

    // model parameters
    sim.a       = parser("-a"      ).asDouble(0.002);
    sim.cs1     = parser("-cs1"    ).asDouble(30);
    sim.cs2     = parser("-cs2"    ).asDouble(40);
    sim.b1      = parser("-b1"     ).asDouble(4500);
    sim.b2      = parser("-b2"     ).asDouble(7000);
    sim.rm      = parser("-rm"     ).asDouble(0.003);
    sim.Anc     = parser("-Anc"    ).asDouble(1);
    sim.epsilon = parser("-epsilon").asDouble(0.3);
    sim.sigmab  = parser("-sigmab" ).asDouble(5.67e-8);
    sim.H       = parser("-H"      ).asDouble(2);
    sim.rhos    = parser("-rhos"   ).asDouble(700);
    sim.rhog    = parser("-rhog"   ).asDouble(1);
    sim.cps     = parser("-cps"    ).asDouble(1800);
    sim.cpg     = parser("-cpg"    ).asDouble(1043);
    sim.A1      = parser("-A1"     ).asDouble(22e5);
    sim.A2      = parser("-A2"     ).asDouble(2e7);
    sim.Dbuoyx  = parser("-Dbuoyx" ).asDouble(0.8);
    sim.Dbuoyy  = parser("-Dbuoyy" ).asDouble(1);
    sim.Ad      = parser("-Ad"     ).asDouble(0.125);

    // initial condition parameters
    sim.initialConditions.Ta      = parser("-Ta"     ).asDouble(300);

    sim.initialConditions.number_of_zones = parser("-zones").asInt(1);
    for (int i = 0; i < sim.initialConditions.number_of_zones; i++)
    {
      sim.initialConditions.Ti.push_back            (parser("-Ti"             + std::to_string(i)).asDouble(1200) );
      sim.initialConditions.xignition.push_back     (parser("-xignition"      + std::to_string(i)).asDouble(100)  );
      sim.initialConditions.yignition.push_back     (parser("-yignition"      + std::to_string(i)).asDouble(250)  );
      sim.initialConditions.xside_ignition.push_back(parser("-xside_ignition" + std::to_string(i)).asDouble(20)   );
      sim.initialConditions.yside_ignition.push_back(parser("-yside_ignition" + std::to_string(i)).asDouble(30)   );
    }

    sim.initialConditions.number_of_roads = parser("-roads").asInt(0);
    for (int i = 0; i < sim.initialConditions.number_of_zones; i++)
    {
      sim.initialConditions.Troad.push_back     (parser("-Troad"      + std::to_string(i)).asDouble(300));
      sim.initialConditions.xroad.push_back     (parser("-xroad"      + std::to_string(i)).asDouble(500));
      sim.initialConditions.yroad.push_back     (parser("-yroad"      + std::to_string(i)).asDouble(500));
      sim.initialConditions.xside_road.push_back(parser("-xside_road" + std::to_string(i)).asDouble(500));
      sim.initialConditions.yside_road.push_back(parser("-yside_road" + std::to_string(i)).asDouble(20) );
    }

    // velocity field parameters
    sim.z0     = parser("-z0"   ).asDouble(0.25);
    sim.delta  = parser("-delta").asDouble(0.04);
    sim.eta    = parser("-eta"  ).asDouble(3);
    sim.kappa  = parser("-kappa").asDouble(0.41);
    sim.u10x   = parser("-u10x" ).asDouble(5);
    sim.u10y   = parser("-u10y" ).asDouble(5);

    // fuel initial conditions
    sim.S1min = parser("-S1min").asDouble(0.15);
    sim.S2min = parser("-S2min").asDouble(0.85);
    sim.S1max = parser("-S1max").asDouble(sim.S1min);
    sim.S2max = parser("-S2max").asDouble(sim.S2min);
    sim.ic_seed = parser("-ic_seed").asInt(0);

    // constants that depend on the above parameters
    sim.C2 = sim.a*sim.A1/sim.cps;
    sim.C3 = sim.a*sim.A2/sim.cps;
    sim.C4 = 1.0/(sim.H*sim.rhos*sim.cps);
    sim.gamma = sim.cpg/sim.cps;
    sim.lambda = sim.rhog/sim.rhos;

    // (constant) velocity field, derived from all parameters above
    const double dx     = (sim.H - sim.z0) - sim.delta*sim.u10x;
    const double dy     = (sim.H - sim.z0) - sim.delta*sim.u10y;
    const double uxstar = sim.u10x * sim.kappa / log((10-dx)/sim.z0);
    const double uystar = sim.u10y * sim.kappa / log((10-dy)/sim.z0);
    const double uhx    = uxstar/sim.kappa*log((sim.H-dx)/sim.z0);
    const double uhy    = uystar/sim.kappa*log((sim.H-dy)/sim.z0);
    const double coef   = (1.0-exp(-sim.eta)) / sim.eta;
    sim.ux = uhx*coef;
    sim.uy = uhy*coef;
    if (sim.verbose) std::cout << "[WS2D] Velocity Field: ux = " << sim.ux << " uy = " << sim.uy << std::endl;
  }

  //3. Allocate grids
  if(sim.verbose) std::cout << "[WS2D] Allocating Grid..." << std::endl;
  sim.allocateGrid();

  //4. Create compute pipeline
  if(sim.verbose) std::cout << "[WS2D] Creating Computational Pipeline..." << std::endl;
  pipeline.push_back(std::make_shared<AdaptTheMesh>(sim));
  pipeline.push_back(std::make_shared<advDiff>(sim));

  //5. Impose initial conditions and initial grid refinement
  if(sim.verbose) std::cout << "[WS2D] Imposing Initial Conditions..." << std::endl;
  IC ic(sim);
  ic(0);
  for (int i = 0 ; i < sim.levelMax; i++)
  {
    (*pipeline[0])(0);
    ic(0);
  }

  if(sim.verbose)
  {
    std::cout << "[WS2D] Operator ordering:\n";
    for (size_t c=0; c<pipeline.size(); c++)
      std::cout << "[WS2D] - " << pipeline[c]->getName() << "\n";
  }
}

Simulation::~Simulation() {}

void Simulation::simulate()
{
  if (sim.verbose) std::cout << "[WS2D] Starting Simulation..." << std::endl;

  //Loop where timesteps are performed, until termination
  while (1)
	{
    //1. Compute current timestep
    const double dt = calcMaxTimestep();

    //2. Print some information on screen
    if (sim.verbose) printf("[WS2D] step:%d, blocks per rank:%zu, time:%f dt:%f\n",sim.step,sim.T->getBlocksInfo().size(),sim.time,sim.dt);
         
    //3. Save fields, if needed
    if( sim.bDump() )
    {
      if(sim.verbose) std::cout << "[WS2D] dumping fields...\n";
      sim.nextDumpTime += sim.dumpTime;
      sim.dumpAll("wildfires_");
    }

    //4. Execute the operators that make up one timestep
    for (size_t c=0; c<pipeline.size(); c++) (*pipeline[c])(dt);
    sim.time += dt;
    sim.step++;

    //5. Save some quantities of interest (if any)
    if(sim.rank == 0)
    {
      std::stringstream ssF;
      ssF<<sim.path4serialization<<"/qoi.dat";
      std::stringstream & fout = logger.get_stream(ssF.str());
      if(sim.step==0)
       fout<<"t dt \n";
      fout<<sim.time<<" "<<sim.dt<<" \n";
    }

    //6. Check if simulation should terminate
    if (sim.bOver())
    {
      if( sim.bDump() )
      {
        if(sim.verbose) std::cout << "[WS2D] dumping fields...\n";
        sim.nextDumpTime += sim.dumpTime;
        sim.dumpAll("wildfires_");
      }
      break;
    }
  }

  if (sim.verbose)
  {
    std::cout << "[WS2D] Simulation Over... Profiling information:\n";
    sim.printResetProfiler();
  }
}

double Simulation::calcMaxTimestep()
{
  const double eps = 1.0;
  const double h = sim.getH();
  const double uMax = sqrt(sim.ux*sim.ux+sim.uy*sim.uy);
  const double dtAdvection = sim.CFL * h / ( uMax + 1e-8 ); //assuming C1/C0 = 1

  //To compute the current maximum timestep, we first need to find the dispersion coefficients.
  //This is done in the steps outlined below.

  //1. Find maximum temperature
  const std::vector<BlockInfo>& TInfo = sim.T->getBlocksInfo();
  double Tmax = -1;
  #pragma omp parallel for reduction(max:Tmax)
  for (size_t i=0; i < TInfo.size(); i++)
  {
    auto & T  = (*sim.T)(i);
    for(int iy=0; iy<ScalarBlock::sizeY; ++iy)
    for(int ix=0; ix<ScalarBlock::sizeX; ++ix)
    {
      Tmax = std::max(T(ix,iy).s,Tmax);
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &Tmax, 1, MPI_DOUBLE, MPI_MAX, sim.comm);

  //2. Find all N locations (xi,yi) with temperature T(xi,yi) close to Tmax : |T(xi,yi)-Tmax| < eps.
  //   Then average all N points to get location of Tmax: 
  //   xmax = sum(i=1,...,N) xi/N
  //   ymax = sum(i=1,...,N) yi/N
  double xmax = 0;
  double ymax = 0;
  int nnn = 0;
  #pragma omp parallel for reduction (+:xmax,ymax,nnn)
  for (size_t i=0; i < TInfo.size(); i++)
  {
    auto & T  = (*sim.T)(i);
    for(int iy=0; iy<ScalarBlock::sizeY; ++iy)
    for(int ix=0; ix<ScalarBlock::sizeX; ++ix)
    {
      const bool isclose = std::fabs(T(ix,iy).s-Tmax) < eps;
      if (isclose)
      {
        double p[2];
        TInfo[i].pos(p,ix,iy);
        nnn ++;
        xmax += p[0];
        ymax += p[1];
      }
    }
  }
  double txy[3] = {xmax,ymax,(double)nnn};
  MPI_Allreduce(MPI_IN_PLACE, txy, 3, MPI_DOUBLE, MPI_SUM, sim.comm);
  xmax = txy[0]/txy[2];
  ymax = txy[1]/txy[2];

  //3. Find the location closest to the maximum temperature's location with T=Ttarget
  /*
     To do so, we look for all locations with |T(x,y)-Ttarget| < eps (we arbitrarily set eps=1.0)
     If we only find one location, we keep that.
     If we find multiple locations, we keep the one that is closer to xmax/ymax
     This means we do this two times, one for x and one for y.
  */
  double Ttarget = 0.1*Tmax + sim.initialConditions.Ta;
  double deltaTx = 1e10;
  double deltaTy = 1e10;
  double dxmin   = 1e10;
  double dymin   = 1e10;
  double xtarget = 1e10;
  double ytarget = 1e10;
  const int signx = sim.ux > 0 ? 1:-1;
  const int signy = sim.uy > 0 ? 1:-1;
  #pragma omp parallel
  {
    double mydeltaTx = 1e10;
    double mydeltaTy = 1e10;
    double mydxmin   = 1e10;
    double mydymin   = 1e10;
    double myxtarget = 1e10;
    double myytarget = 1e10;
    #pragma omp for
    for (size_t i=0; i < TInfo.size(); i++)
    {
      auto & T  = (*sim.T)(i);
      for(int iy=0; iy<ScalarBlock::sizeY; ++iy)
      for(int ix=0; ix<ScalarBlock::sizeX; ++ix)
      {
        double p[2];
        TInfo[i].pos(p,ix,iy);
        const double dx = signx*(p[0]-xmax) > 0 ? std::fabs(p[0]-xmax) : 1e10;
        const double dy = signy*(p[1]-ymax) > 0 ? std::fabs(p[1]-ymax) : 1e10;
        const double dT = std::fabs(T(ix,iy).s-Ttarget);

        if ( (std::fabs(mydeltaTx-dT) < eps || dT < mydeltaTx) && dx < mydxmin )
        {
          mydxmin = dx;
          mydeltaTx = std::min(dT,mydeltaTx);
          myxtarget = p[0];
        }
        if ( (std::fabs(mydeltaTy-dT) < eps || dT < mydeltaTy) && dy < mydymin )
        {
          mydymin = dy;
          mydeltaTy = std::min(dT,mydeltaTy);
          myytarget = p[1];
        }
      }
    }
    #pragma omp critical
    {
      if ( ( mydeltaTx < deltaTx) && (mydxmin < dxmin) )
      {
        dxmin = mydxmin;
        deltaTx = mydeltaTx;
        xtarget = myxtarget;
      }
      if ( (mydeltaTy < deltaTy) && (mydymin < dymin) )
      {
        dymin = mydymin;
        deltaTy = mydeltaTy;
        ytarget = myytarget;
      }
    }
  }

  double Lcx = std::fabs(xmax-xtarget);
  double Lcy = std::fabs(ymax-ytarget);

  //now keep the minimum Lc among ranks
  MPI_Allreduce(MPI_IN_PLACE, &Lcx, 1, MPI_DOUBLE, MPI_MIN, sim.comm);
  MPI_Allreduce(MPI_IN_PLACE, &Lcy, 1, MPI_DOUBLE, MPI_MIN, sim.comm);

  if (sim.verbose) std::cout << "   characteristic fire lengths:" << Lcx << "," << Lcy << std::endl;

  sim.Deffx = sim.Dbuoyx + sim.Ad * sim.ux * Lcx;
  sim.Deffy = sim.Dbuoyy + sim.Ad * sim.uy * Lcy;

  
  //Now that we found the dispersion coefficients we can compute the maximum allowed timestep
  const double dtDiffusion = 0.25*h*h/(sim.Deffx+sim.Deffy+0.25*h*uMax);
  sim.dt = std::min({ dtDiffusion, dtAdvection});

  return sim.dt;
}
