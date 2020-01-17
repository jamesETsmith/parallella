#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void QR(Eigen::MatrixXd &A, Eigen::MatrixXd &Q, Eigen::MatrixXd &R,
        std::string output_file = "qr.json") {
  /* Modfied Gram-Schmidt for QR decomposition.

  According to Demmel Chpt 3 the modified is more stable that the classical
  Gram-Schmidt
  */

  /* Algorithm 3.1
  for i = 1 to n // compute ith columns of Q and R
    qi = ai
    for j = 1 to i — 1 // subtract component in qj direction from ai
        rji = qj ai //CGS
        rji = qj qi //MGS
        qi = qi — rji qj
    end for

    rii = || qi ||_2
    if rii = 0 // a2 is linearly dependent on al,... ,al
        quit
    end if
    q
     /= rii
  end for
  */

  // Set up json output
  json output;
  // Create standalone output with default name
  // otherwise read in and append existing file
  if (output_file != "qr.json") {
    std::ifstream output_in(output_file);
    output_in >> output;
  }

  Q = Eigen::MatrixXd::Zero(A.rows(), A.cols());
  R = Eigen::MatrixXd::Zero(A.rows(), A.cols());

  for (int i = 0; i < A.cols(); i++) {
    Q.col(i) = A.col(i);

    for (int j = 0; j < i; j++) {
      R(j, i) = Q.col(j).transpose() * Q.col(i);  // MGS
      Q.col(i) -= R(j, i) * Q.col(j);
    }

    R(i, i) = Q.col(i).norm();
    if (R(i, i) == 0) {
      throw std::invalid_argument("A is linearly dependent!!");
    }

    Q.col(i) /= R(i, i);
  }

  // Dump json
  std::ofstream out(output_file);
  out << std::setw(4) << output << std::endl;
}