//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#pragma once

#include "Operator.h"

class AdaptTheMesh : public Operator
{
 public:
  ScalarAMR *  T_amr  = nullptr;
  ScalarAMR * S1_amr  = nullptr;
  ScalarAMR * S2_amr  = nullptr;

  AdaptTheMesh(SimulationData& s) : Operator(s)
  {
    T_amr  = new ScalarAMR(*sim.T ,sim.Rtol,sim.Ctol);
    S1_amr = new ScalarAMR(*sim.S1,sim.Rtol,sim.Ctol);
    S2_amr = new ScalarAMR(*sim.S2,sim.Rtol,sim.Ctol);
  }

  ~AdaptTheMesh()
  {
    if( T_amr   not_eq nullptr ) delete T_amr ;
    if( S1_amr  not_eq nullptr ) delete S1_amr;
    if( S2_amr  not_eq nullptr ) delete S2_amr;
  }

  void operator() (const Real dt) override;
  void adapt();

  std::string getName() override
  {
    return "AdaptTheMesh";
  }
};
