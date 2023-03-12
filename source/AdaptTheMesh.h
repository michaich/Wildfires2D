#pragma once

#include "Operator.h"

class AdaptTheMesh : public Operator
{
 public:
  ScalarAMR *  T_amr  = nullptr;
  ScalarAMR * S1_amr  = nullptr;
  ScalarAMR * S2_amr  = nullptr;
  ScalarAMR * tmp_amr  = nullptr;

  AdaptTheMesh(SimulationData& s) : Operator(s)
  {
    T_amr   = new ScalarAMR(*sim.T  ,1.0,0.01);
    S1_amr  = new ScalarAMR(*sim.S1 ,1.0,0.01);
    S2_amr  = new ScalarAMR(*sim.S2 ,1.0,0.01);
    tmp_amr = new ScalarAMR(*sim.tmp,1.0,0.01);
  }

  ~AdaptTheMesh()
  {
    if( T_amr   not_eq nullptr ) delete T_amr ;
    if( S1_amr  not_eq nullptr ) delete S1_amr;
    if( S2_amr  not_eq nullptr ) delete S2_amr;
    if( tmp_amr  not_eq nullptr ) delete tmp_amr;
  }

  void operator() (const Real dt) override;
  void adapt();

  std::string getName() override
  {
    return "AdaptTheMesh";
  }
};
