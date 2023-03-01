//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "Operator.h"
#include "Cubism/FluxCorrection.h"

class IC : public Operator
{
  public:
  IC(SimulationData& s) : Operator(s) { }

  void operator()(const Real dt);

  std::string getName() {
    return "IC";
  }
};