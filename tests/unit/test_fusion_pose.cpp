#include "fusion_pose.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <optional>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

/// Smooth quadratic height field in world coordinates:
///   z = a*x^2 + b*x*y + c*y^2 + d*x + e*y + f.
/// Curvature in both image directions makes all six ICP degrees of freedom
/// observable; a single plane would leave tangential motion unconstrained.
struct Surface {
    double a{0.30};
    double b{0.20};
    double c{0.35};
    double d{-0.15};
    double e{0.10};
    double f{1.50};
};

std::array<double, 9> rotation_from_pose(const std::array<float, 12> &pose) {
    return {static_cast<double>(pose[0]), static_cast<double>(pose[1]), static_cast<double>(pose[2]),
            static_cast<double>(pose[4]), static_cast<double>(pose[5]), static_cast<double>(pose[6]),
            static_cast<double>(pose[8]), static_cast<double>(pose[9]), static_cast<double>(pose[10])};
}

std::array<double, 9> rotation_axis_angle(double ax, double ay, double az, double angle_rad) {
    const double norm = std::sqrt(ax * ax + ay * ay + az * az);
    ax /= norm;
    ay /= norm;
    az /= norm;
    const double cosine = std::cos(angle_rad);
    const double sine = std::sin(angle_rad);
    const double t = 1.0 - cosine;
    return {
        t * ax * ax + cosine,    t * ax * ay - sine * az, t * ax * az + sine * ay,
        t * ax * ay + sine * az, t * ay * ay + cosine,    t * ay * az - sine * ax,
        t * ax * az - sine * ay, t * ay * az + sine * ax, t * az * az + cosine,
    };
}

std::array<float, 12> pose_from_rotation(const std::array<double, 9> &rotation, double tx, double ty, double tz) {
    return {static_cast<float>(rotation[0]), static_cast<float>(rotation[1]), static_cast<float>(rotation[2]),
            static_cast<float>(tx),          static_cast<float>(rotation[3]), static_cast<float>(rotation[4]),
            static_cast<float>(rotation[5]), static_cast<float>(ty),          static_cast<float>(rotation[6]),
            static_cast<float>(rotation[7]), static_cast<float>(rotation[8]), static_cast<float>(tz)};
}

std::array<float, 12> identity_pose() {
    return {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F};
}

/// Smallest positive ray parameter s with o + s*d on the surface, or nullopt.
std::optional<double> intersect_surface(const Surface &surface, double ox, double oy, double oz, double dx, double dy,
                                        double dz) {
    const double a2 = surface.a * dx * dx + surface.b * dx * dy + surface.c * dy * dy;
    const double a1 = 2.0 * surface.a * ox * dx + surface.b * (ox * dy + oy * dx) + 2.0 * surface.c * oy * dy +
                      surface.d * dx + surface.e * dy - dz;
    const double a0 = surface.a * ox * ox + surface.b * ox * oy + surface.c * oy * oy + surface.d * ox +
                      surface.e * oy + surface.f - oz;

    auto first_positive = [](double root0, double root1) -> std::optional<double> {
        const double lo = std::min(root0, root1);
        const double hi = std::max(root0, root1);
        if (lo > 0.0) {
            return lo;
        }
        if (hi > 0.0) {
            return hi;
        }
        return std::nullopt;
    };

    if (std::abs(a2) < 1e-12) {
        if (std::abs(a1) < 1e-12) {
            return std::nullopt;
        }
        return first_positive(-a0 / a1, -a0 / a1);
    }
    const double discriminant = a1 * a1 - 4.0 * a2 * a0;
    if (discriminant < 0.0) {
        return std::nullopt;
    }
    const double root = std::sqrt(discriminant);
    return first_positive((-a1 - root) / (2.0 * a2), (-a1 + root) / (2.0 * a2));
}

/// Render the analytic surface from a w2c camera. Stored values are metric
/// depth in the current camera (s along the ray (xn, yn, 1)); 0 marks invalid.
cv::Mat render_depth(const Surface &surface, const da3::fusion::Pinhole &intrinsics,
                     const std::array<float, 12> &pose_w2c, int width, int height) {
    cv::Mat depth(height, width, CV_32F, cv::Scalar(0.0));
    const std::array<double, 9> rotation = rotation_from_pose(pose_w2c);
    const double tx = static_cast<double>(pose_w2c[3]);
    const double ty = static_cast<double>(pose_w2c[7]);
    const double tz = static_cast<double>(pose_w2c[11]);

    // Camera centre C = -R^T t.
    const double camera_x = -(rotation[0] * tx + rotation[3] * ty + rotation[6] * tz);
    const double camera_y = -(rotation[1] * tx + rotation[4] * ty + rotation[7] * tz);
    const double camera_z = -(rotation[2] * tx + rotation[5] * ty + rotation[8] * tz);

    for (int v = 0; v < height; ++v) {
        float *row = depth.ptr<float>(v);
        for (int u = 0; u < width; ++u) {
            const double xn =
                (static_cast<double>(u) - static_cast<double>(intrinsics.cx)) / static_cast<double>(intrinsics.fx);
            const double yn =
                (static_cast<double>(v) - static_cast<double>(intrinsics.cy)) / static_cast<double>(intrinsics.fy);
            // World direction of the current-camera ray (xn, yn, 1).
            const double dx = rotation[0] * xn + rotation[3] * yn + rotation[6];
            const double dy = rotation[1] * xn + rotation[4] * yn + rotation[7];
            const double dz = rotation[2] * xn + rotation[5] * yn + rotation[8];
            const std::optional<double> s = intersect_surface(surface, camera_x, camera_y, camera_z, dx, dy, dz);
            if (s && *s >= 0.3 && *s <= 3.0) {
                row[u] = static_cast<float>(*s);
            }
        }
    }
    return depth;
}

double translation_error(const std::array<float, 12> &estimate, const std::array<float, 12> &truth) {
    const double dx = static_cast<double>(estimate[3]) - static_cast<double>(truth[3]);
    const double dy = static_cast<double>(estimate[7]) - static_cast<double>(truth[7]);
    const double dz = static_cast<double>(estimate[11]) - static_cast<double>(truth[11]);
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double rotation_error_deg(const std::array<float, 12> &estimate, const std::array<float, 12> &truth) {
    const std::array<double, 9> a = rotation_from_pose(estimate);
    const std::array<double, 9> b = rotation_from_pose(truth);
    // R_rel = A * B^T. Use the antisymmetric part (first order in the error
    // angle) rather than a trace, which loses all precision below ~0.01 deg
    // once the matrices are rounded to float.
    std::array<double, 9> relative{};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            relative[i * 3 + j] =
                a[i * 3 + 0] * b[j * 3 + 0] + a[i * 3 + 1] * b[j * 3 + 1] + a[i * 3 + 2] * b[j * 3 + 2];
        }
    }
    const double skew_x = relative[7] - relative[5];
    const double skew_y = relative[2] - relative[6];
    const double skew_z = relative[3] - relative[1];
    const double sine_angle = 0.5 * std::sqrt(skew_x * skew_x + skew_y * skew_y + skew_z * skew_z);
    return std::asin(std::clamp(sine_angle, -1.0, 1.0)) * 180.0 / kPi;
}

void print_result(const char *label, const std::array<float, 12> &recovered, const std::array<float, 12> &truth) {
    std::cout << std::setprecision(9) << "[  INFO ] " << label << ": recovered t = (" << recovered[3] << ", "
              << recovered[7] << ", " << recovered[11] << ") m, truth t = (" << truth[3] << ", " << truth[7] << ", "
              << truth[11] << ") m, |dt| = " << translation_error(recovered, truth)
              << " m, rotation error = " << rotation_error_deg(recovered, truth) << " deg\n";
}

} // namespace

TEST(FusionPose, IcpRecoversPureTranslation) {
    constexpr int kWidth = 160;
    constexpr int kHeight = 120;
    const da3::fusion::Pinhole intrinsics{200.0F, 200.0F, 80.0F, 60.0F};
    const Surface surface;

    const std::array<float, 12> reference_pose = identity_pose();
    const std::array<float, 12> truth = pose_from_rotation(rotation_axis_angle(0.0, 0.0, 1.0, 0.0), 0.02, -0.01, 0.03);

    const cv::Mat reference = render_depth(surface, intrinsics, reference_pose, kWidth, kHeight);
    const cv::Mat current = render_depth(surface, intrinsics, truth, kWidth, kHeight);
    ASSERT_GT(cv::countNonZero(reference > 0.0), kWidth * kHeight / 2);
    ASSERT_GT(cv::countNonZero(current > 0.0), kWidth * kHeight / 2);

    da3::fusion::IcpPoseProvider provider({reference, current}, intrinsics, reference_pose);
    std::array<float, 12> seed{};
    ASSERT_TRUE(provider.next(seed));
    std::array<float, 12> recovered{};
    ASSERT_TRUE(provider.next(recovered));
    EXPECT_TRUE(provider.has_real_tracking());

    print_result("pure translation", recovered, truth);
    EXPECT_LT(translation_error(recovered, truth), 1e-3);
    EXPECT_LT(rotation_error_deg(recovered, truth), 0.1);

    std::array<float, 12> exhausted{};
    EXPECT_FALSE(provider.next(exhausted));
}

TEST(FusionPose, IcpRecoversSmallRotation) {
    constexpr int kWidth = 160;
    constexpr int kHeight = 120;
    const da3::fusion::Pinhole intrinsics{200.0F, 200.0F, 80.0F, 60.0F};
    const Surface surface;

    const std::array<float, 12> reference_pose = identity_pose();
    const std::array<float, 12> truth =
        pose_from_rotation(rotation_axis_angle(0.3, -0.5, 0.8, 3.0 * kPi / 180.0), 0.0, 0.0, 0.0);

    const cv::Mat reference = render_depth(surface, intrinsics, reference_pose, kWidth, kHeight);
    const cv::Mat current = render_depth(surface, intrinsics, truth, kWidth, kHeight);
    ASSERT_GT(cv::countNonZero(reference > 0.0), kWidth * kHeight / 2);
    ASSERT_GT(cv::countNonZero(current > 0.0), kWidth * kHeight / 2);

    da3::fusion::IcpPoseProvider provider({reference, current}, intrinsics, reference_pose);
    std::array<float, 12> seed{};
    ASSERT_TRUE(provider.next(seed));
    std::array<float, 12> recovered{};
    ASSERT_TRUE(provider.next(recovered));

    print_result("small rotation", recovered, truth);
    EXPECT_LT(translation_error(recovered, truth), 1e-3);
    EXPECT_LT(rotation_error_deg(recovered, truth), 0.1);
}

TEST(FusionPose, IcpIdentityConvergesToIdentity) {
    constexpr int kWidth = 160;
    constexpr int kHeight = 120;
    const da3::fusion::Pinhole intrinsics{200.0F, 200.0F, 80.0F, 60.0F};
    const Surface surface;

    const std::array<float, 12> identity = identity_pose();
    const cv::Mat depth = render_depth(surface, intrinsics, identity, kWidth, kHeight);
    ASSERT_GT(cv::countNonZero(depth > 0.0), kWidth * kHeight / 2);

    da3::fusion::IcpPoseProvider provider({depth, depth}, intrinsics, identity);
    std::array<float, 12> seed{};
    ASSERT_TRUE(provider.next(seed));
    std::array<float, 12> recovered{};
    ASSERT_TRUE(provider.next(recovered));

    print_result("identity", recovered, identity);
    EXPECT_LT(translation_error(recovered, identity), 1e-6);
    EXPECT_LT(rotation_error_deg(recovered, identity), 1e-4);
}

TEST(FusionPose, IcpComposesWithPreviousPose) {
    constexpr int kWidth = 160;
    constexpr int kHeight = 120;
    const da3::fusion::Pinhole intrinsics{200.0F, 200.0F, 80.0F, 60.0F};
    const Surface surface;

    const std::array<float, 12> first_pose =
        pose_from_rotation(rotation_axis_angle(0.2, 0.9, 0.3, 4.0 * kPi / 180.0), 0.05, -0.03, 0.10);
    const da3::fusion::detail::Pose3d relative =
        da3::fusion::detail::compose(da3::fusion::detail::pose_from_array(pose_from_rotation(
                                         rotation_axis_angle(0.0, 1.0, 0.0, 1.0 * kPi / 180.0), 0.01, -0.005, 0.02)),
                                     da3::fusion::detail::pose_from_array(first_pose));
    const std::array<float, 12> truth = da3::fusion::detail::pose_to_array(relative);

    const cv::Mat reference = render_depth(surface, intrinsics, first_pose, kWidth, kHeight);
    const cv::Mat current = render_depth(surface, intrinsics, truth, kWidth, kHeight);
    ASSERT_GT(cv::countNonZero(reference > 0.0), kWidth * kHeight / 2);
    ASSERT_GT(cv::countNonZero(current > 0.0), kWidth * kHeight / 2);

    da3::fusion::IcpPoseProvider provider({reference, current}, intrinsics, first_pose);
    std::array<float, 12> seed{};
    ASSERT_TRUE(provider.next(seed));
    std::array<float, 12> recovered{};
    ASSERT_TRUE(provider.next(recovered));

    print_result("composed with seed pose", recovered, truth);
    EXPECT_LT(translation_error(recovered, truth), 1e-3);
    EXPECT_LT(rotation_error_deg(recovered, truth), 0.1);
}

TEST(FusionPose, IcpRejectsInvalidDepth) {
    constexpr int kWidth = 160;
    constexpr int kHeight = 120;
    const da3::fusion::Pinhole intrinsics{200.0F, 200.0F, 80.0F, 60.0F};
    const Surface surface;
    const std::array<float, 12> identity = identity_pose();

    const cv::Mat reference = render_depth(surface, intrinsics, identity, kWidth, kHeight);
    const cv::Mat invalid = cv::Mat::zeros(kHeight, kWidth, CV_32F);
    ASSERT_GT(cv::countNonZero(reference > 0.0), kWidth * kHeight / 2);

    da3::fusion::IcpPoseProvider provider({reference, invalid}, intrinsics, identity);
    std::array<float, 12> seed{};
    ASSERT_TRUE(provider.next(seed));
    std::array<float, 12> rejected{};
    EXPECT_FALSE(provider.next(rejected));

    // The failure latches: a later call must not report a stale pose.
    EXPECT_FALSE(provider.next(rejected));
}

TEST(FusionPose, IcpRejectsInconsistentOutlierScene) {
    constexpr int kWidth = 160;
    constexpr int kHeight = 120;
    const da3::fusion::Pinhole intrinsics{200.0F, 200.0F, 80.0F, 60.0F};
    const Surface surface;
    const std::array<float, 12> identity = identity_pose();

    const cv::Mat reference = render_depth(surface, intrinsics, identity, kWidth, kHeight);
    // A fronto-parallel wall 0.5 m closer than the reference surface: every
    // correspondence is a distance outlier, so the frame must be rejected
    // rather than sliding the wall into place.
    const cv::Mat outlier_wall(kHeight, kWidth, CV_32F, cv::Scalar(1.0));
    ASSERT_GT(cv::countNonZero(reference > 0.0), kWidth * kHeight / 2);

    da3::fusion::IcpPoseProvider provider({reference, outlier_wall}, intrinsics, identity);
    std::array<float, 12> seed{};
    ASSERT_TRUE(provider.next(seed));
    std::array<float, 12> rejected{};
    EXPECT_FALSE(provider.next(rejected));
}

TEST(FusionPose, ExternalProviderSequencesPoses) {
    const std::array<float, 12> first = identity_pose();
    const std::array<float, 12> second = pose_from_rotation(rotation_axis_angle(0.0, 0.0, 1.0, 0.2), 0.1, 0.2, 0.3);

    da3::fusion::ExternalPoseProvider provider({first, second});
    EXPECT_TRUE(provider.has_real_tracking());

    std::array<float, 12> pose{};
    ASSERT_TRUE(provider.next(pose));
    EXPECT_EQ(pose, first);
    ASSERT_TRUE(provider.next(pose));
    EXPECT_EQ(pose, second);
    EXPECT_FALSE(provider.next(pose));
}

TEST(FusionPose, ModelProviderRequiresModelPoses) {
    ::tasks::FrameData with_pose;
    with_pose.pose = pose_from_rotation(rotation_axis_angle(0.1, 0.2, 0.3, 0.1), 0.01, 0.02, 0.03);
    with_pose.pose_from_model = true;

    ::tasks::FrameData without_pose;
    without_pose.pose = identity_pose();
    without_pose.pose_from_model = false;

    da3::fusion::ModelPoseProvider mixed({with_pose, without_pose});
    EXPECT_FALSE(mixed.has_real_tracking());
    std::array<float, 12> pose{};
    ASSERT_TRUE(mixed.next(pose));
    EXPECT_EQ(pose, with_pose.pose);
    // The identity fallback must not be emitted as if it were tracked.
    EXPECT_FALSE(mixed.next(pose));

    da3::fusion::ModelPoseProvider tracked({with_pose, with_pose});
    EXPECT_TRUE(tracked.has_real_tracking());
    ASSERT_TRUE(tracked.next(pose));
    EXPECT_EQ(pose, with_pose.pose);
    ASSERT_TRUE(tracked.next(pose));
    EXPECT_EQ(pose, with_pose.pose);
    EXPECT_FALSE(tracked.next(pose));

    da3::fusion::ModelPoseProvider empty(std::vector<::tasks::FrameData>{});
    EXPECT_FALSE(empty.has_real_tracking());
    EXPECT_FALSE(empty.next(pose));
}
