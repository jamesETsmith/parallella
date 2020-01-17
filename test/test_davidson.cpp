#define CATCH_CONFIG_MAIN
#define EIGEN_USE_MKL_ALL

#include <stdlib.h>
#include <Eigen/Dense>
#include <catch.hpp>

#include <parallella.hpp>

Eigen::MatrixXd make_diagonally_dominant_mat(
    const int N, const double off_diag_scale = 0.001) {
  Eigen::MatrixXd tmp = off_diag_scale * Eigen::MatrixXd::Random(N, N);
  Eigen::VectorXd tmp2 = Eigen::VectorXd::LinSpaced(N, 0, N);
  Eigen::MatrixXd diag_shift = tmp2.asDiagonal();
  Eigen::MatrixXd A = tmp + tmp.transpose() + diag_shift;
  return A;
}

TEST_CASE("Davidson Liu", "[davidson]") {
  SECTION("small") {
    INFO("Running Small Test");
    // Set up control
    srand(20);
    const int N = 20;
    auto A = make_diagonally_dominant_mat(N);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);

    // Use parallella
    const int n_roots = 5;
    auto lamb = Davidson_Liu(A, n_roots, 1e-8, "test1.json");
    const double lamb_error = (es.eigenvalues().head(n_roots) - lamb).norm();

    REQUIRE(lamb_error < 1e-6);
  }

  SECTION("medium") {
    INFO("Running Medium Test");
    // Set up control
    srand(20);
    const int N = 100;
    auto A = make_diagonally_dominant_mat(N, 0.01);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);

    // Use parallella
    const int n_roots = 5;
    auto lamb = Davidson_Liu(A, n_roots, 1e-8, "test2.json");
    const double lamb_error = (es.eigenvalues().head(n_roots) - lamb).norm();

    REQUIRE(lamb_error < 1e-6);
  }

  SECTION("large") {
    INFO("Running Large Test");

    // Set up control
    srand(20);
    const int N = 150;
    auto A = make_diagonally_dominant_mat(N, 0.1);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);

    // Use parallella
    const int n_roots = 5;
    auto lamb = Davidson_Liu(A, n_roots, 1e-8, "test3.json");
    const double lamb_error = (es.eigenvalues().head(n_roots) - lamb).norm();

    REQUIRE(lamb_error < 1e-6);
  }
}
