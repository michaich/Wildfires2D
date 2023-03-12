#include "Helpers.h"
#include <random>
#include <fstream>
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

  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
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
        double p[2];
        TInfo[i].pos(p,ix,iy);
        S1(ix,iy).s = distributionS1(generator);
        S2(ix,iy).s = distributionS2(generator);
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
