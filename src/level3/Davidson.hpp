#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

Eigen::VectorXd Davidson_Liu(Eigen::MatrixXd &A, const int n_roots = 5,
                             const double eigenvalue_tol = 1e-8,
                             std::string output_file = "davidson.json") {
  /*
    Use the Davidson method to solve or multiple roots simultaneously.

    See Sherrill 1999 ADVANCES IN QUANTUM CHEMISTRY, VOLUME 34. for more
    details. Equations numbers are from fig 5.
  */
  std::cout << std::setprecision(3);
  auto start_total = std::chrono::system_clock::now();

  int L = n_roots;
  const int N = A.rows();
  const int max_subspace_size = std::min(N, 20 * L);
  const int max_iter = 100;

  // Set up json output
  json output;
  // Create standalone output with default name
  // otherwise read in and append existing file
  if (output_file != "davidson.json") {
    std::ifstream output_in(output_file);
    output_in >> output;
  }
  output["davidson"]["max subspace size"] = max_subspace_size;
  output["davidson"]["subspace size"] = std::vector<int>(0);
  output["davidson"]["lambda"] = std::vector<std::vector<double>>(0);
  output["davidson"]["delta lambda"] = std::vector<double>(0);
  output["davidson"]["iter timing (s)"] = std::vector<double>(0);
  output["davidson"]["converged"] = false;

  // Step 1
  Eigen::MatrixXd b = Eigen::MatrixXd::Zero(max_subspace_size, N);
  b.topLeftCorner(n_roots, n_roots) =
      Eigen::MatrixXd::Identity(n_roots, n_roots);
  Eigen::MatrixXd delta = Eigen::MatrixXd::Zero(n_roots, N);
  Eigen::VectorXd lamb_old = Eigen::VectorXd::Zero(n_roots);

  for (int iter = 0; iter < max_iter; iter++) {
    // Logging
    output["davidson"]["subspace size"].push_back(L);
    auto iter_start = std::chrono::system_clock::now();

    // Step 2
    Eigen::MatrixXd G =
        b.block(0, 0, L, N) * A * b.block(0, 0, L, N).transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(G);
    Eigen::VectorXd lamb = es.eigenvalues().head(n_roots);
    Eigen::MatrixXd alpha = es.eigenvectors().block(0, 0, G.rows(), n_roots);
    output["davidson"]["lambda"].push_back(lamb);

    // Step 3
    Eigen::MatrixXd delta = Eigen::MatrixXd::Zero(n_roots, N);
    for (int k = 0; k < n_roots; k++) {
      Eigen::MatrixXd I = Eigen::MatrixXd::Identity(N, N);
      Eigen::MatrixXd temp =
          (A - lamb(k) * I) * b.block(0, 0, L, N).transpose();
      Eigen::VectorXd rk = temp * alpha.col(k);
      delta.row(k) =
          rk.array() /
          (lamb(k) * Eigen::VectorXd::Ones(N) - A.diagonal()).array();

    }  // end for(k)

    // Step 4
    for (auto dk : delta.rowwise()) {
      dk /= dk.norm();
    }

    // Step 5
    int m = 0;  // Keep track of how many
    for (auto dk : delta.rowwise()) {
      if (m > L) {
        break;
      }
      // Orthoganlize dk vectors w.r.t. all trial vectors {b_i}
      for (int i = 0; i < L; i++) {
        dk -= dk.dot(b.row(i)) * b.row(i);
      }  // end for i

      if (dk.norm() > 1e-3) {
        b.row(L) = dk / dk.norm();
        L++;
      }
      m++;
    }  // end for(dk)

    // convergence check
    if (L >= N) {
      // Crash
      output["davidson"]["converged"] = false;
      exit(1);
    }

    double delta_lamb = (lamb - lamb_old).norm();
    if (delta_lamb < eigenvalue_tol) {
      output["davidson"]["delta lambda"].push_back(delta_lamb);
      output["davidson"]["converged"] = true;

      auto iter_end = std::chrono::system_clock::now();
      std::chrono::duration<double> iter_time_s = iter_end - iter_start;
      output["davidson"]["iter timing (s)"].push_back(iter_time_s.count());
      break;

    } else {
      output["davidson"]["delta lambda"].push_back(delta_lamb);

      auto iter_end = std::chrono::system_clock::now();
      std::chrono::duration<double> iter_time_s = iter_end - iter_start;
      output["davidson"]["iter timing (s)"].push_back(iter_time_s.count());
    }
    // End convergence check

    lamb_old = lamb;

  }  // end for iter

  // Wrap up timing
  auto end_total = std::chrono::system_clock::now();
  std::chrono::duration<double> total_time_s = end_total - start_total;
  output["davidson"]["total time (s)"] = total_time_s.count();

  // Dump json
  std::ofstream out(output_file);
  out << std::setw(4) << output << std::endl;

  return lamb_old;
}