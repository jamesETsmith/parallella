#include <Eigen/Dense>

Eigen::VectorXd inner_product(Eigen::VectorXd a, Eigen::VectorXd b) {
  return a.adjoint() * b;
}