#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include "rb10_rmpflow_rviz/casadi_rmp2_solver.hpp"

int main(int argc, char ** argv)
{
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: generate_casadi_rmp2_policy URDF_PATH OUTPUT_BASE [MAX_OBSTACLES]\n";
    return 2;
  }

  try {
    rb10_rmpflow_rviz::EigenRmpConfig config;
    config.solve_method = "rmp2";
    config.casadi_rmp2_max_obstacles = argc == 4 ? std::stoi(argv[3]) : 24;
    config.casadi_rmp2_jit = false;
    rb10_rmpflow_rviz::CasadiRmp2Solver solver(config, argv[1]);
    const std::string generated = solver.generate_c_code(argv[2]);
    std::cout << generated << '\n';
    return 0;
  } catch (const std::exception & error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
