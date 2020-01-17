#include <chrono>
#include <iostream>

#include <parallella.hpp>

Eigen::MatrixXd make_diagonally_dominant_mat(
    const int N, const double off_diag_scale = 0.001) {
  Eigen::MatrixXd tmp = off_diag_scale * Eigen::MatrixXd::Random(N, N);
  Eigen::VectorXd tmp2 = Eigen::VectorXd::LinSpaced(N, 0, N);
  Eigen::MatrixXd diag_shift = tmp2.asDiagonal();
  Eigen::MatrixXd A = tmp + tmp.transpose() + diag_shift;
  return A;
}

int main(int argc, char** argv) {
  auto start_total = std::chrono::system_clock::now();

  // Set up control
  srand(20);
  const int N = 2000;
  auto A = make_diagonally_dominant_mat(N, 0.01);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);

  // Use parallella
  const int n_roots = 5;
  auto lamb = Davidson_Liu(A, n_roots, 1e-8);
  const double lamb_error = (es.eigenvalues().head(n_roots) - lamb).norm();
  std::cout << "Error = " << lamb_error << std::endl;

  auto end_total = std::chrono::system_clock::now();
  std::chrono::duration<double> total_time_s = end_total - start_total;
  std::cout << "TOTAL TIME = " << total_time_s.count() << " (s)" << std::endl;
  return 0;
}