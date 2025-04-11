#include "accumulator.h" 
#include <cmath>
#include <Eigen/SVD>

RotationAccumulator::RotationAccumulator()
    : theta_shift_buf_(Vector3d::Zero()),
    theta_dipole_buf_(Vector3d::Zero()),
    theta_diffuse_buf_(Vector3d::Zero()),
    xyz_shift_buf_(Vector3d::Zero()),
    xyz_dipole_buf_(Vector3d::Zero()),
    xyz_diffuse_buf_(Vector3d::Zero()),
    R_total_(Matrix3d::Identity()) {}

void RotationAccumulator::addDipole(const Eigen::Vector<double, 6>& p) {
    theta_dipole_buf_ += p.tail<3>();
    xyz_dipole_buf_ += p.head<3>();
    maybeApplyDipole();
}

void RotationAccumulator::addShift(const Eigen::Vector<double, 6>& p) {
    theta_shift_buf_ += p.tail<3>();
    xyz_shift_buf_ += p.head<3>();
    maybeApplyShift();
}

void RotationAccumulator::addDiffuse(const Eigen::Vector<double, 6>& p) {
    theta_diffuse_buf_ += p.tail<3>();
    xyz_diffuse_buf_ += p.head<3>();
    maybeApplyDiffuse();
}

void RotationAccumulator::maybeApplyShift() {
    //printf("dx: %f\n", theta_shift_buf_.norm());
    if (theta_shift_buf_.norm() > shift_thresh_) {
        R_total_ = ExpSO3(theta_shift_buf_) * R_total_;
        theta_shift_buf_.setZero();
        if (svd_count % svd_max == 0) {
            Eigen::JacobiSVD<Matrix3d> svd(R_total_, Eigen::ComputeFullU | Eigen::ComputeFullV);
            R_total_ = svd.matrixU() * svd.matrixV().transpose();
            svd_count = 0;
        }
        svd_count++;
    }
}

void RotationAccumulator::maybeApplyDiffuse() {
    //if (theta_diffuse_buf_.norm() > 0.2)
     //   printf("aaa\n");
    if (theta_diffuse_buf_.norm() > diffuse_thresh_) {
        R_total_ = ExpSO3(theta_diffuse_buf_) * R_total_;
        theta_diffuse_buf_.setZero();
        if (svd_count % svd_max == 0) {
            Eigen::JacobiSVD<Matrix3d> svd(R_total_, Eigen::ComputeFullU | Eigen::ComputeFullV);
            R_total_ = svd.matrixU() * svd.matrixV().transpose();
            svd_count = 0;
        }
        svd_count++;
    }
}

void RotationAccumulator::maybeApplyDipole() {
    //printf("dr: %f\n", theta_dipole_buf_.norm());
    if (theta_dipole_buf_.norm() > dipole_thresh_) {
        R_total_ = ExpSO3(theta_dipole_buf_) * R_total_;
        theta_dipole_buf_.setZero();
        if (svd_count % svd_max == 0) {
            Eigen::JacobiSVD<Matrix3d> svd(R_total_, Eigen::ComputeFullU | Eigen::ComputeFullV);
            R_total_ = svd.matrixU() * svd.matrixV().transpose();
            svd_count = 0;
        }
        svd_count++;
    }
}

Vector3d RotationAccumulator::applyTo(const Vector3d& P) const {
    return P + xyz_diffuse_buf_ + xyz_dipole_buf_ + xyz_shift_buf_;
}

Matrix3d RotationAccumulator::applyTo(const Matrix3d& U) const {
    return R_total_ * U;
}

Matrix3d RotationAccumulator::ExpSO3(const Vector3d& theta) {
    double angle = theta.norm();
    Matrix3d K;
    K << 0, -theta.z(), theta.y(),
        theta.z(), 0, -theta.x(),
        -theta.y(), theta.x(), 0;

    Matrix3d I = Matrix3d::Identity();

    if (angle < 1e-8) {
        return I + K + 0.5 * K * K;
    }
    else {
        return I + (sin(angle) / angle) * K
            + ((1 - cos(angle)) / (angle * angle)) * K * K;
    }
}
