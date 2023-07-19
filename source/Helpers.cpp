#include "Helpers.h"
#include <random>
#include <fstream>
using namespace cubism;

void IC::operator()(const double dt)
{
  const std::vector<BlockInfo>& TInfo = sim.T->getBlocksInfo();

  const double Ta = sim.initialConditions.Ta;

  #pragma omp parallel
  {
    #ifdef USEOPENMP
    const int tid = omp_get_thread_num();
    #else
    const int tid = 0;
    #endif
    std::default_random_engine generator (sim.ic_seed + tid);
    std::uniform_real_distribution<double> distributionS1 (sim.S1min,sim.S1max);
    std::uniform_real_distribution<double> distributionS2 (sim.S2min,sim.S2max);
    #pragma omp for
    for (size_t i=0; i < TInfo.size(); i++)
    {
  
      auto & T  = (*sim.T)(i);
      auto & S1 = (*sim.S1)(i);
      auto & S2 = (*sim.S2)(i);
      for(int iy=0; iy<ScalarBlock::sizeY; ++iy)
      for(int ix=0; ix<ScalarBlock::sizeX; ++ix)
      {
        S1(ix,iy).s = distributionS1(generator);
        S2(ix,iy).s = distributionS2(generator);
        T(ix,iy).s = Ta;


        double p[2];
        TInfo[i].pos(p,ix,iy);
        const double x = p[0];
        const double y = p[1];

        //check if we are in a road or an ingition zone
        for (int n = 0; n < sim.initialConditions.number_of_zones; n++)
        {
          const double Ti      = sim.initialConditions.Ti            [n];
          const double xcenter = sim.initialConditions.xignition     [n];
          const double ycenter = sim.initialConditions.yignition     [n];
          const double xside   = sim.initialConditions.xside_ignition[n];
          const double yside   = sim.initialConditions.yside_ignition[n];
          if ( std::fabs(x-xcenter) < 0.5*xside && std::fabs(y-ycenter) < 0.5*yside)
          {
            T(ix,iy).s = Ti;
          }
        }
        for (int n = 0; n < sim.initialConditions.number_of_roads; n++)
        {
          const double Ti      = sim.initialConditions.Troad     [n];
          const double xcenter = sim.initialConditions.xroad     [n];
          const double ycenter = sim.initialConditions.yroad     [n];
          const double xside   = sim.initialConditions.xside_road[n];
          const double yside   = sim.initialConditions.yside_road[n];
          if ( std::fabs(x-xcenter) < 0.5*xside && std::fabs(y-ycenter) < 0.5*yside)
          {
            T(ix,iy).s = Ti;
            S1(ix,iy).s = 0.0;
            S2(ix,iy).s = 0.0;
          }
        }
      }
    }

    #pragma omp for
    for (size_t i=0; i < TInfo.size(); i++)
    {
      auto & S2_0 = (*sim.S2_0)(i);
      const auto & S2 = (*sim.S2)(i);
      for(int iy=0; iy<ScalarBlock::sizeY; ++iy)
      for(int ix=0; ix<ScalarBlock::sizeX; ++ix)
      {
        S2_0(ix,iy).s = S2(ix,iy).s;
      }
    }
  }
}

BufferedLogger logger;

static constexpr int AUTO_FLUSH_COUNT = 100;

void BufferedLogger::flush(BufferedLogger::container_type::iterator it) {
    std::ofstream savestream;
    savestream.open(it->first, std::ios::app | std::ios::out);
    savestream << it->second.stream.rdbuf();
    savestream.close();
    it->second.requests_since_last_flush = 0;
}

std::stringstream& BufferedLogger::get_stream(const std::string &filename) {
    auto it = files.find(filename);
    if (it != files.end()) {
        if (++it->second.requests_since_last_flush == AUTO_FLUSH_COUNT)
            flush(it);
        return it->second.stream;
    } else {
        // With request_since_last_flush == 0,
        // the first flush will have AUTO_FLUSH_COUNT frames.
        auto new_it = files.emplace(filename, Stream()).first;
        return new_it->second.stream;
    }
}
