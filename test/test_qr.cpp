#define CATCH_CONFIG_MAIN
#define EIGEN_USE_MKL_ALL

#include <stdlib.h>
#include <Eigen/Dense>
#include <catch.hpp>

#include <parallella.hpp>

TEST_CASE("QR", "decomp") {
  srand(20);

  SECTION("small") {
    INFO("Running Small Test");
    const int N = 20;

    Eigen::MatrixXd A = Eigen::MatrixXd::Random(N, N);
    A = A.transpose() * A;

    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;

    QR(A, Q, R);

    Eigen::MatrixXd Ap = Q * R;
    auto error = (Ap - A).norm();
    REQUIRE(error < 1e-6);
  }

  SECTION("medium") {
    INFO("Running Small Test");
    const int N = 200;

    Eigen::MatrixXd A = Eigen::MatrixXd::Random(N, N);
    A = A.transpose() * A;

    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;

    QR(A, Q, R);

    Eigen::MatrixXd Ap = Q * R;
    auto error = (Ap - A).norm();
    REQUIRE(error < 1e-6);
  }

  SECTION("large") {
    INFO("Running Large Test");
    const int N = 1000;

    Eigen::MatrixXd A = Eigen::MatrixXd::Random(N, N);
    A = A.transpose() * A;

    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;

    QR(A, Q, R);

    Eigen::MatrixXd Ap = Q * R;
    auto error = (Ap - A).norm();
    REQUIRE(error < 1e-6);
  }
}