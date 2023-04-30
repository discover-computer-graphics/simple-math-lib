use crate::{Vec3, Vec4};
use std::ops::{Index, IndexMut, Mul};
use std::ptr::{slice_from_raw_parts, slice_from_raw_parts_mut};

/// A 4x4 matrix with 32 bits floating components.
///
/// # Examples
/// ```
/// use simple_math_lib::Mat4;
/// let m1 = Mat4::new();
/// let m2 = Mat4::identity();
/// assert_eq!(m1, m2);
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Mat4 {
    x: Vec4,
    y: Vec4,
    z: Vec4,
    w: Vec4,
}

/// Constants
impl Mat4 {
    /// Identity matrix.
    pub const IDENTITY: Self = Self::identity();

    /// The zero matrix.
    pub const ZERO: Self = Self::splat(0.0);

    /// Creates a new identity matrix.
    #[inline(always)]
    pub const fn identity() -> Self {
        Mat4 {
            x: Vec4::new(1.0, 0.0, 0.0, 0.0),
            y: Vec4::new(0.0, 1.0, 0.0, 0.0),
            z: Vec4::new(0.0, 0.0, 1.0, 0.0),
            w: Vec4::new(0.0, 0.0, 0.0, 1.0),
        }
    }

    /// Creates a new identity matrix.
    #[inline(always)]
    pub const fn new() -> Self {
        Self::identity()
    }

    /// Creates a new zero matrix.
    #[inline(always)]
    pub const fn zero() -> Self {
        Mat4 {
            x: Vec4::ZERO,
            y: Vec4::ZERO,
            z: Vec4::ZERO,
            w: Vec4::ZERO,
        }
    }

    /// Creates a new matrix from values.
    #[allow(clippy::too_many_arguments)]
    #[rustfmt::skip]
    #[inline(always)]
    pub const fn from_values(
        n00: f32, n01: f32, n02: f32, n03: f32,
        n10: f32, n11: f32, n12: f32, n13: f32,
        n20: f32, n21: f32, n22: f32, n23: f32,
        n30: f32, n31: f32, n32: f32, n33: f32,
    ) -> Self {
        Mat4 {
            x: Vec4::new(n00, n10, n20, n30),
            y: Vec4::new(n01, n11, n21, n31),
            z: Vec4::new(n02, n12, n22, n32),
            w: Vec4::new(n03, n13, n23, n33),
        }
    }

    /// Creates a new matrix from column vectors.
    #[inline(always)]
    pub const fn from_vectors(x: Vec4, y: Vec4, z: Vec4, w: Vec4) -> Self {
        Mat4 { x, y, z, w }
    }

    /// Creates a new matrix with all elements set to `n`.
    #[inline(always)]
    pub const fn splat(n: f32) -> Self {
        Mat4 {
            x: Vec4::splat(n),
            y: Vec4::splat(n),
            z: Vec4::splat(n),
            w: Vec4::splat(n),
        }
    }
}

impl Mat4 {
    /// Creates a new orthographic projection matrix.
    #[inline]
    pub fn new_orthographic(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let w_inv = 1.0 / (right - left);
        let h_inv = 1.0 / (top - bottom);
        let d_inv = 1.0 / (far - near);

        #[rustfmt::skip]
        return Mat4::from_values(
            2.0 * w_inv,         0.0,   0.0, -(right + left) * w_inv,
                    0.0, 2.0 * h_inv,   0.0, -(top + bottom) * h_inv,
                    0.0,         0.0, d_inv,           -near * d_inv,
                    0.0,         0.0,   0.0,                     1.0,
        );
    }

    /// Creates a new perspective projection matrix in a right-handed coordinate system.
    #[inline]
    pub fn new_perspective(fov_y: f32, aspect_ratio: f32, near: f32, far: f32) -> Self {
        let g = 1.0 / (fov_y * 0.5).tan();
        let k = 1.0 / (near - far);
        let a = g / aspect_ratio;
        let b = (near + far) * k;
        let c = 2.0 * near * far * k;

        #[rustfmt::skip]
        return Mat4::from_values(
              a, 0.0,  0.0, 0.0,
            0.0,   g,  0.0, 0.0,
            0.0, 0.0,    b,   c,
            0.0, 0.0, -1.0, 0.0,
        );
    }

    /// Creates a new camera transform positioned at `eye` and looking
    /// in the direction `direction` moving everything from world space to camera space.
    #[inline]
    pub fn new_look_to(eye: Vec3, direction: Vec3, up: Vec3) -> Self {
        let v = direction.normalize();
        let r = v.cross(up).normalize();
        let u = r.cross(v);

        #[rustfmt::skip]
        return Mat4::from_values(
             r.x,  r.y,  r.z, -eye.dot(r),
             u.x,  u.y,  u.z, -eye.dot(u),
            -v.x, -v.y, -v.z,  eye.dot(v),
             0.0,  0.0,  0.0,             1.0,
        );
    }

    /// Creates a new camera transform positioned at `eye` and looking at `target`
    /// moving everything from world space to camera space.
    #[inline]
    pub fn new_look_at(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        Self::new_look_to(eye, target - eye, up)
    }

    /// Creates a new affine transform from a translation.
    #[inline]
    pub fn new_translation(translation: impl Into<Vec3>) -> Self {
        let mut result = Mat4::identity();
        result.w = translation.into().to_homogeneous_point();
        result
    }

    /// Creates a new linear transformation that represents a rotation
    /// through an angle in radian about the X axis.
    #[inline]
    pub fn new_rotation_x(angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();

        #[rustfmt::skip]
        return Mat4::from_values(
            1.0, 0.0, 0.0, 0.0,
            0.0,   c,  -s, 0.0,
            0.0,   s,   c, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );
    }

    /// Creates a new linear transformation that represents a rotation
    /// through an angle in radian about the Y axis.
    #[inline]
    pub fn new_rotation_y(angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();

        #[rustfmt::skip]
        return Mat4::from_values(
              c, 0.0,   s, 0.0,
            0.0, 1.0,  -s, 0.0,
             -s, 0.0,   c, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );
    }

    /// Creates a new linear transformation that represents a rotation
    /// through an angle in radian about the X axis.
    #[inline]
    pub fn new_rotation_z(angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();

        #[rustfmt::skip]
        return Mat4::from_values(
              c,  -s, 0.0, 0.0,
              s,   c, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );
    }

    /// Creates a new linear transformation that represents a rotation
    /// by an angle in radian about an arbitrary axis.
    #[inline]
    pub fn new_rotation(angle: f32, axis: Vec3) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        let d = 1.0 - c;

        let x = axis.x * d;
        let y = axis.y * d;
        let z = axis.z * d;

        let axay = x * axis.y;
        let axaz = x * axis.z;
        let ayaz = y * axis.z;

        #[rustfmt::skip]
        return Mat4::from_values(
               c + x * axis.x, axay - s * axis.z, axaz + s * axis.y, 0.0,
            axay + s * axis.z,    c + y * axis.y, ayaz - s * axis.x, 0.0,
            axaz - s * axis.y, ayaz + s * axis.x,    c + z * axis.z, 0.0,
                          0.0,               0.0,               0.0, 1.0,
        );
    }

    /// Creates a new scaling transformation with factors along axis X, Y and Z.
    #[inline]
    pub fn new_scale(scale_x: f32, scale_y: f32, scale_z: f32) -> Self {
        #[rustfmt::skip]
        return Mat4::from_values(
            scale_x,     0.0,     0.0, 0.0,
                0.0, scale_y,     0.0, 0.0,
                0.0,     0.0, scale_z, 0.0,
                0.0,     0.0,     0.0, 1.0,
        );
    }

    /// Creates a new scaling transformation with a `scaling_factor` along an arbitrary `direction`.
    #[inline]
    pub fn new_scale_along_arbitrary_direction(scaling_factor: f32, direction: Vec3) -> Self {
        let s = scaling_factor - 1.0;
        let x = direction.x * s;
        let y = direction.y * s;
        let z = direction.z * s;
        let dx_dy = x * direction.y;
        let dx_dz = x * direction.z;
        let dy_dz = y * direction.z;

        #[rustfmt::skip]
        return Mat4::from_values(
            x * direction.x + 1.0,                 dx_dy,                 dx_dz, 0.0,
                            dx_dy, y * direction.y + 1.0,                 dy_dz, 0.0,
                            dx_dz,                 dy_dz, z * direction.z + 1.0, 0.0,
                              0.0,                   0.0,                   0.0, 1.0,
        );
    }

    /// Creates a new shear transformation that represents a skew by an angle
    /// along the direction `a` based on the projected length along the direction `b`.
    ///
    /// `a` and `b` parameters are assumed to be unit length.
    #[inline]
    pub fn new_skew(angle: f32, a: Vec3, b: Vec3) -> Self {
        let t = angle.tan();
        let x = a.x * t;
        let y = a.y * t;
        let z = a.z * t;

        #[rustfmt::skip]
        return Mat4::from_values(
            x * b.x + 1.0,       x * b.y,       x * b.z, 0.0,
                  y * b.x, y * b.y + 1.0,       y * b.z, 0.0,
                  z * b.x,       z * b.y, z * b.z + 1.0, 0.0,
                      0.0,           0.0,           0.0, 1.0,
        );
    }
}

impl Mat4 {
    /// Returns the translation of the affine transformation.
    #[inline]
    pub fn get_translation(self) -> Vec3 {
        self.w.into()
    }

    /// Defines the translations of the affine transformation.
    #[inline]
    pub fn with_translation(mut self, t: Vec3) -> Self {
        self.w = t.to_homogeneous_point();
        self
    }

    /// Checks if the matrix is homogeneous.
    #[inline]
    pub fn is_homogeneous(self) -> bool {
        self[(3, 0)] == 0.0 && self[(3, 1)] == 0.0 && self[(3, 2)] == 0.0 && self[(3, 3)] == 1.0
    }

    /// Converts a matrix to homogeneous coordinates by setting the last row to `[0, 0, 0, 1]`.
    #[inline]
    pub fn to_homogeneous(mut self) -> Self {
        self[(3, 0)] = 0.0;
        self[(3, 1)] = 0.0;
        self[(3, 2)] = 0.0;
        self[(3, 3)] = 1.0;
        self
    }

    /// Returns a slice containing all components in column-major order.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        unsafe { &*slice_from_raw_parts(self.x.as_ptr(), 16) }
    }

    /// Returns a mutable slice containing all components in column-major order.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { &mut *slice_from_raw_parts_mut(self.x.as_mut_ptr(), 16) }
    }

    /// Returns a raw pointer to the underlying data of the matrix.
    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        self.x.as_ptr()
    }

    /// Returns a mutable pointer to the underlying data of the matrix.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.x.as_mut_ptr()
    }
}

impl Index<usize> for Mat4 {
    type Output = Vec4;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("index out of bounds"),
        }
    }
}

impl IndexMut<usize> for Mat4 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("index out of bounds"),
        }
    }
}

impl Index<(usize, usize)> for Mat4 {
    type Output = f32;

    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self[index.1][index.0]
    }
}

impl IndexMut<(usize, usize)> for Mat4 {
    #[inline]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self[index.1][index.0]
    }
}

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;

    #[inline]
    fn mul(self, rhs: Vec4) -> Self::Output {
        self[0] * rhs[0] + self[1] * rhs[1] + self[2] * rhs[2] + self[3] * rhs[3]
    }
}

impl Mul for Mat4 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Mat4::from_vectors(self * rhs[0], self * rhs[1], self * rhs[2], self * rhs[3])
    }
}
