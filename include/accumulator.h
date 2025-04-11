#pragma once
#include <Eigen/Dense>
using Eigen::Vector3d;
using Eigen::Matrix3d;

class RotationAccumulator {
public:
    RotationAccumulator();

    void addDipole(const Eigen::Vector<double, 6>& p);
    void addShift(const Eigen::Vector<double, 6>& p);
    void addDiffuse(const Eigen::Vector<double, 6>& p);
    Vector3d applyTo(const Vector3d& P) const;
    Matrix3d applyTo(const Matrix3d& U) const;
    const Matrix3d& rotation() const { return R_total_; }
    const Vector3d& position() const { 
        return xyz_diffuse_buf_ + xyz_shift_buf_ + xyz_dipole_buf_; }

private:
    Vector3d theta_shift_buf_;
    Vector3d theta_dipole_buf_;
    Vector3d theta_diffuse_buf_;
    Vector3d xyz_shift_buf_;
    Vector3d xyz_dipole_buf_;
    Vector3d xyz_diffuse_buf_;
    Matrix3d R_total_;
    

    static Matrix3d ExpSO3(const Vector3d& theta);
    void maybeApplyShift();
    void maybeApplyDipole();
    void maybeApplyDiffuse();

    const double shift_thresh_= 1e-4;
    const double dipole_thresh_= 1e-4;
    const double diffuse_thresh_ = 1e-4;

    const int svd_max = 1;
    int svd_count = 0;
};
