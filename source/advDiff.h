#pragma once

#include "Operator.h"
#include "Cubism/FluxCorrection.h"

class advDiff : public Operator
{
 public:
  advDiff(SimulationData& s) : Operator(s) { }

  std::vector<double> Trhs;
  std::vector<double> S1rhs;
  std::vector<double> S2rhs;

  void operator() (const Real dt) override;

  std::string getName() override
  {
    return "advDiff";
  }
};