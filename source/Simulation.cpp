#include "Simulation.h"

#include <Cubism/HDF5Dumper.h>
#include "Helpers.h"
#include "advDiff.h"
#include "AdaptTheMesh.h"
#include "BufferedLogger.h"
#include <algorithm>
#include <iterator>

using namespace cubism;

BCflag cubismBCX;
BCflag cubismBCY;

static const char kHorLine[] = "================================================================\n";

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
    sim.endTime    = parser("-tend"      ).asDouble(10.)          ; //simulation ends at t=tend (inactive if tend=0)
  
    // Boundary conditions (freespace or periodic)
    std::string BC_x = parser("-BC_x").asString("freespace");
    std::string BC_y = parser("-BC_y").asString("freespace");
    cubismBCX = string2BCflag(BC_x);
    cubismBCY = string2BCflag(BC_y);
  
    // output parameters
    sim.dumpTime           = parser("-tdump"        ).asDouble(0.1);
    sim.path2file          = parser("-file"         ).asString("./");
    sim.path4serialization = parser("-serialization").asString(sim.path2file);
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
    sim.Ta      = parser("-Ta"     ).asDouble(300);
    sim.Ti      = parser("-Ti"     ).asDouble(1200);
    sim.xcenter = parser("-xcenter").asDouble(100);
    sim.ycenter = parser("-ycenter").asDouble(250);
    sim.xside   = parser("-xside"  ).asDouble(20);
    sim.yside   = parser("-yside"  ).asDouble(30);
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
}


void Simulation::simulate()
{
  if (sim.verbose) std::cout << kHorLine << "[WS2D] Starting Simulation..." << std::endl;

  //Loop where timesteps are performed, until termination
  while (1)
	{
    //1. Compute current timestep
    const double dt = calcMaxTimestep();

    //2. Print some information on screen
    if (sim.verbose)
    {
      std::cout << kHorLine;
      printf("[WS2D] step:%d, blocks:%zu, time:%f",sim.step,sim.T->getBlocksInfo().size(),sim.time);
    }
         
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
      if(sim.verbose)
        std::cout << "[WS2D] running " << pipeline[c]->getName() << "...\n";
      (*pipeline[c])(dt);
    }
    sim.time += dt;
    sim.step++;

    if(sim.rank == 0)
    {
      std::stringstream ssF;
      ssF<<sim.path2file<<"/qoi.dat";
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
    std::cout << kHorLine << "[WS2D] Simulation Over... Profiling information:\n";
    sim.printResetProfiler();
    std::cout << kHorLine;
  }
}

double Simulation::calcMaxTimestep()
{
  sim.dt = 1.0;
  //const double h = sim.getH();
  //const auto findMaxU = findMaxU(sim);
  //const double uMax = findMaxU.run();
  //const double dtDiffusion = 0.25*h*h/(sim.nu+0.25*h*uMax);
  //const double dtAdvection = h / ( uMax + 1e-8 );
  //sim.dt = std::min({ dtDiffusion, CFL * dtAdvection});
  //if( sim.dt <= 0 ){
  //  std::cout << "[CUP2D] dt <= 0. Aborting..." << std::endl;
  //  fflush(0);
  //  abort();
  //}
  return sim.dt;
}
