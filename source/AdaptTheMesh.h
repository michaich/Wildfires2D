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
    T_amr  = new ScalarAMR(*sim.T ,sim.Rtol,sim.Ctol);
    S1_amr = new ScalarAMR(*sim.S1,sim.Rtol,sim.Ctol);
    S2_amr = new ScalarAMR(*sim.S2,sim.Rtol,sim.Ctol);
    tmp_amr = new ScalarAMR(*sim.tmp,sim.Rtol,sim.Ctol);
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
