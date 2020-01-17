#include "VV.hpp"

int main(int argc, char** argv) {
  Eigen::VectorXd a;
  a << 0, 1, 2, 3, 4;
  Eigen::VectorXd b;
  b << 1, 1, 2, 3, 4;

  Eigen::VectorXd c = inner_product(a, b);
  return 0;
}