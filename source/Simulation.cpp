#include "Simulation.h"

#include <Cubism/HDF5Dumper.h>
#include "Helpers.h"
#include "advDiff.h"
#include "AdaptTheMesh.h"
#include <algorithm>
#include <iterator>

using namespace cubism;

static void getmaxT(double * invec, double * inoutvec, int *len, MPI_Datatype *dtype)
{
  if (invec[0] > inoutvec[0])
  {
    for (int i=0; i<*len; i++ ) inoutvec[i] = invec[i];
  }
}

Simulation::Simulation(int argc, char ** argv, MPI_Comm comm) : parser(argc,argv)
{
  //1. Print initial general information
  {
    sim.comm = comm;
    int size;
    MPI_Comm_size(sim.comm,&size);
    MPI_Comm_rank(sim.comm,&sim.rank);
    if (sim.rank == 0)
    {
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
    sim.Rtol     = parser("-Rtol"    ).asDouble(1000); //tolerance for mesh refinement
    sim.Ctol     = parser("-Ctol"    ).asDouble(400); //tolerance for mesh compression

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
    sim.initialConditions.Ti      = parser("-Ti"     ).asDouble(1200);
    sim.initialConditions.xcenter = parser("-xcenter").asDouble(100);
    sim.initialConditions.ycenter = parser("-ycenter").asDouble(250);
    sim.initialConditions.xside   = parser("-xside"  ).asDouble(20);
    sim.initialConditions.yside   = parser("-yside"  ).asDouble(30);

    // velocity field parameters
    sim.velocityField.z0     = parser("-z0"    ).asDouble(0.25);
    sim.velocityField.lambda = parser("-lambda").asDouble(0.04);
    sim.velocityField.alpha  = parser("-alpha" ).asDouble(3);
    sim.velocityField.kappa  = parser("-kappa" ).asDouble(0.41);
    sim.velocityField.u10x   = parser("-u10x"  ).asDouble(5);
    sim.velocityField.u10y   = parser("-u10y"  ).asDouble(5);

    // constants that depend on the above parameters
    sim.C2 = sim.a*sim.A1/sim.cps;
    sim.C3 = sim.a*sim.A2/sim.cps;
    sim.C4 = 1.0/(sim.H*sim.rhos*sim.cps);
    sim.gamma = sim.cpg/sim.cps;
    sim.lambda = sim.rhog/sim.rhos;

    // (constant) velocity field, derived from all parameters above
    const double z0 = sim.velocityField.z0;
    const double kappa = sim.velocityField.kappa;
    const double u10x = sim.velocityField.u10x;
    const double u10y = sim.velocityField.u10y;
    const double lambda = sim.velocityField.lambda;
    const double alpha = sim.velocityField.alpha;
    const double H = sim.H;

    const double dx = (H - z0) - lambda*u10x;
    const double dy = (H - z0) - lambda*u10y;

    const double uxstar = u10x * kappa / log((10-dx)/z0);
    const double uystar = u10y * kappa / log((10-dy)/z0);

    const double uhx = uxstar/kappa*log((H-dx)/z0);
    const double uhy = uystar/kappa*log((H-dy)/z0);

    const double coef = (1.0-exp(-alpha)) / alpha; 
    sim.ux = uhx*coef;
    sim.uy = uhy*coef;
    if (sim.verbose) std::cout << "[WS2D] Velocity Field: ux = " << sim.ux << " uy = " << sim.uy << std::endl;
  }

  //3. Allocate grids
  if(sim.verbose) std::cout << "[WS2D] Allocating Grid..." << std::endl;
  sim.allocateGrid();

  //4. Impose initial conditions
  if(sim.verbose) std::cout << "[WS2D] Imposing Initial Conditions..." << std::endl;
  IC ic(sim);
  ic(0);

  //5. Create compute pipeline
  if(sim.verbose)
    std::cout << "[WS2D] Creating Computational Pipeline..." << std::endl;
  pipeline.push_back(std::make_shared<AdaptTheMesh>(sim));
  pipeline.push_back(std::make_shared<advDiff>(sim));
  if(sim.verbose)
  {
    std::cout << "[WS2D] Operator ordering:\n";
    for (size_t c=0; c<pipeline.size(); c++)
      std::cout << "[WS2D] - " << pipeline[c]->getName() << "\n";
  }

  //6. Create custom MPI max (for finding max temperature and positions)
  MPI_Op_create( (MPI_User_function *)getmaxT, 1, &custom_max );
}

Simulation::~Simulation()
{
  MPI_Op_free( &custom_max );
}

void Simulation::simulate()
{
  if (sim.verbose) std::cout << "[WS2D] Starting Simulation..." << std::endl;

  //Loop where timesteps are performed, until termination
  while (1)
	{
    //1. Compute current timestep
    const double dt = calcMaxTimestep();

    //2. Print some information on screen
    if (sim.verbose)
      printf("[WS2D] step:%d, blocks per rank:%zu, time:%f dt:%f\n",sim.step,sim.T->getBlocksInfo().size(),sim.time,sim.dt);
         
    //3. Save fields, if needed
    if( sim.bDump() )
    {
      if(sim.verbose)
        std::cout << "[WS2D] dumping fields...\n";
      sim.nextDumpTime += sim.dumpTime;
      sim.dumpAll("wildfires_");
    }

    //4. Execute the operators that make up one timestep
    for (size_t c=0; c<pipeline.size(); c++)
    {
      (*pipeline[c])(dt);
    }
    sim.time += dt;
    sim.step++;

    if(sim.rank == 0)
    {
      std::stringstream ssF;
      ssF<<sim.path4serialization<<"/qoi.dat";
      std::stringstream & fout = logger.get_stream(ssF.str());
      if(sim.step==0)
       fout<<"t dt \n";
      fout<<sim.time<<" "<<sim.dt<<" \n";
    }

    //5. Check if simulation should terminate
    if (sim.bOver())
    {
      if( sim.bDump() )
      {
        if(sim.verbose)
          std::cout << "[WS2D] dumping fields...\n";
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
  const double h = sim.getH();
  const double uMax = sqrt(sim.ux*sim.ux+sim.uy*sim.uy);
  const double dtAdvection = sim.CFL * h / ( uMax + 1e-8 ); //assuming C1/C0 = 1


  //First, compute current dispersion coefficients

  //1. Find location with maximum temperature
  const std::vector<BlockInfo>& TInfo = sim.T->getBlocksInfo();
  double Tmax = -1;
  double xmax = -1;
  double ymax = -1;
  #pragma omp parallel
  {
    double myTmax = -1;
    double myxmax =  0;
    double myymax =  0;
    #pragma omp for
    for (size_t i=0; i < TInfo.size(); i++)
    {
      auto & T  = (*sim.T)(i);
      for(int iy=0; iy<ScalarBlock::sizeY; ++iy)
      for(int ix=0; ix<ScalarBlock::sizeX; ++ix)
      {
        if (T(ix,iy).s > myTmax)
        {
          double p[2];
          TInfo[i].pos(p,ix,iy);
          myxmax = p[0];
          myymax = p[1];
          myTmax = T(ix,iy).s;
        }
      }
    }
    #pragma omp critical
    {
      if (myTmax > Tmax)
      {
        Tmax = myTmax;
        xmax = myxmax;
        ymax = myymax;
      }
    }
  }
  double txy[3] = {Tmax,xmax,ymax};
  MPI_Allreduce(MPI_IN_PLACE, txy, 3, MPI_DOUBLE, custom_max, sim.comm);
  Tmax = txy[0];
  xmax = txy[1];
  ymax = txy[2];

  //2. Find the location closest to the maximum temperature's location with T=Ttarget
  /*
     To do so, we look for all locations with |T(x,y)-Ttarget| < eps (we arbitrarily set eps=1.0)
     If we only find one location, we keep that.
     If we find multiple locations, we keep the one that is closer to xmax/ymax
     This means we do this two times, one for x and one for y.
  */
  double Ttarget = 0.1*Tmax + sim.initialConditions.Ta;
  double deltaTx = 1e10;
  double deltaTy = 1e10;
  double eps     = 1.0;
  double dxmin   = 1e10;
  double dymin   = 1e10;
  double xtarget = 1e10;
  double ytarget = 1e10;
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
        const double dx = std::fabs(p[0]-xmax);
        const double dy = std::fabs(p[1]-ymax);
        const double dT = std::fabs(T(ix,iy).s-Ttarget);

        if ( (dT < eps || dT < mydeltaTx) && (dx < mydxmin) )
        {
          mydxmin = dx;
          mydeltaTx = dT;
          myxtarget = p[0];
        }
        if ( (dT < eps || dT < mydeltaTy) && (dy < mydymin) )
        {
          mydymin = dy;
          mydeltaTy = dT;
          myytarget = p[1];
        }
      }
    }

    #pragma omp critical
    {
      if ( (mydeltaTx < eps || mydeltaTx < deltaTx) && (mydxmin < dxmin) )
      {
        dxmin = mydxmin;
        deltaTx = mydeltaTx;
        xtarget = myxtarget;
      }
      if ( (mydeltaTy < eps || mydeltaTy < deltaTy) && (mydymin < dymin) )
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

  double Lcx = 10.0;
  double Lcy = 10.0;

  sim.Deffx = sim.Dbuoyx + sim.Ad * sim.ux * Lcx;
  sim.Deffy = sim.Dbuoyy + sim.Ad * sim.uy * Lcy;

  const double dtDiffusion = 0.25*h*h/(sim.Deffx+sim.Deffy+0.25*h*uMax);
  sim.dt = std::min({ dtDiffusion, dtAdvection});

  return sim.dt;
}
