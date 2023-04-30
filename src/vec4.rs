use std::{
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
    ptr::{slice_from_raw_parts, slice_from_raw_parts_mut},
};

/// Vec4 is a 4 components vector with data stored as 32 bits floating point numbers.
///
/// # Example
/// ```
/// use simple_math_lib::Vec4;
/// let v = Vec4::new(3.0, 6.0, 1.0, 2.0);
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

/// Constants
impl Vec4 {
    /// Vector with all components to 0.
    pub const ZERO: Self = Self::splat(0.0);
    /// Vector with all components to 1.
    pub const ONE: Self = Self::splat(1.0);

    /// Unit vector pointing in the X direction.
    pub const X: Self = Self::new(1.0, 0.0, 0.0, 0.0);
    /// Unit vector pointing in the Y direction.
    pub const Y: Self = Self::new(0.0, 1.0, 0.0, 0.0);
    /// Unit vector pointing in the Z direction.
    pub const Z: Self = Self::new(0.0, 0.0, 1.0, 0.0);
    /// Unit vector pointing in the W direction.
    pub const W: Self = Self::new(0.0, 0.0, 0.0, 1.0);

    /// Creates a new vector from values.
    ///
    /// # Example
    /// ```
    /// use simple_math_lib::Vec4;
    /// let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(1.0, v.x);
    /// assert_eq!(2.0, v.y);
    /// assert_eq!(3.0, v.z);
    /// assert_eq!(4.0, v.w);
    /// ```
    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Vec4 { x, y, z, w }
    }

    /// Creates a new vector with all components set to `v`.
    ///
    /// # Example
    /// ```
    /// use simple_math_lib::Vec4;
    /// let v = Vec4::splat(3.0);
    /// assert_eq!(3.0, v.x);
    /// assert_eq!(3.0, v.y);
    /// assert_eq!(3.0, v.z);
    /// assert_eq!(3.0, v.w);
    /// ```
    #[inline(always)]
    pub const fn splat(v: f32) -> Self {
        Vec4 {
            x: v,
            y: v,
            z: v,
            w: v,
        }
    }
}

impl Vec4 {
    /// Computes the dot product.
    ///
    /// # Example
    /// ```
    /// use simple_math_lib::Vec4;
    /// let v1 = Vec4::new(3.0, 2.0, 4.0, 7.0);
    /// let v2 = Vec4::new(6.0, 1.0, 5.0, 8.0);
    /// assert_eq!(96.0, v1.dot(v2));
    /// ```
    #[inline]
    pub fn dot(self, rhs: Vec4) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }

    /// Gets the magnitude squared of the vector.
    ///
    /// Same as the dot product with itself.
    ///
    /// # Example
    /// ```
    /// use simple_math_lib::Vec4;
    /// let v = Vec4::new(3.0, 4.0, 6.0, 2.0);
    /// assert_eq!(65.0, v.magnitude_squared());
    /// ```
    #[doc(alias = "length")]
    #[inline]
    pub fn magnitude_squared(self) -> f32 {
        self.dot(self)
    }

    /// Gets the magnitude (or length) of the vector.
    ///
    /// Equivalent to the square root of the dot product with itself.
    ///
    /// # Example
    /// ```
    /// use simple_math_lib::Vec4;
    /// let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(30.0f32.sqrt(), v.magnitude());
    /// ```
    #[doc(alias = "length")]
    #[inline]
    pub fn magnitude(self) -> f32 {
        self.magnitude_squared().sqrt()
    }

    /// Normalizes the vector magnitude to 1.
    /// Return the same vector but with a magnitude of 1.
    ///
    /// # Example
    /// ```
    /// use simple_math_lib::Vec4;
    /// let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let v = v.normalize();
    /// assert_eq!(1.0, v.magnitude_squared());
    /// assert_eq!(1.0, v.magnitude());
    /// ```
    #[inline]
    pub fn normalize(self) -> Self {
        let norm = 1.0 / self.magnitude();
        self * norm
    }

    /// Returns a slice containing all the components of the vector.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        unsafe { &*slice_from_raw_parts(&self.x, 4) }
    }

    /// Returns a mutable slice containing all the components of the vector.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { &mut *slice_from_raw_parts_mut(&mut self.x, 4) }
    }

    /// Returns a raw pointer to the first component of the vector.
    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        &self.x as *const f32
    }

    /// Returns a mutable pointer to the first component of the vector.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        &mut self.x as *mut f32
    }
}

impl Index<usize> for Vec4 {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl IndexMut<usize> for Vec4 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}

impl AddAssign for Vec4 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
        self.w += rhs.w;
    }
}

impl Add for Vec4 {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl SubAssign for Vec4 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
        self.w -= rhs.w;
    }
}

impl Sub for Vec4 {
    type Output = Self;

    #[inline]
    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl MulAssign<f32> for Vec4 {
    #[inline]
    fn mul_assign(&mut self, s: f32) {
        self.x *= s;
        self.y *= s;
        self.z *= s;
        self.w *= s;
    }
}

impl Mul<f32> for Vec4 {
    type Output = Self;

    #[inline]
    fn mul(mut self, s: f32) -> Self::Output {
        self *= s;
        self
    }
}

impl DivAssign<f32> for Vec4 {
    #[inline]
    fn div_assign(&mut self, mut s: f32) {
        s = 1.0 / s;
        self.x *= s;
        self.y *= s;
        self.z *= s;
        self.w *= s;
    }
}

impl Div<f32> for Vec4 {
    type Output = Self;

    #[inline]
    fn div(mut self, s: f32) -> Self::Output {
        self /= s;
        self
    }
}

impl Neg for Vec4 {
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self::Output {
        self *= -1.0;
        self
    }
}

impl AsRef<[f32]> for Vec4 {
    #[inline]
    fn as_ref(&self) -> &[f32] {
        self.as_slice()
    }
}

impl AsMut<[f32]> for Vec4 {
    #[inline]
    fn as_mut(&mut self) -> &mut [f32] {
        self.as_mut_slice()
    }
}

impl From<[f32; 4]> for Vec4 {
    #[inline]
    fn from(values: [f32; 4]) -> Self {
        Vec4::new(values[0], values[1], values[2], values[3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vec3;
    use std::ffi::c_void;

    #[test]
    fn new() {
        let v = Vec4::new(2.0, 5.0, 8.0, 3.0);
        assert_eq!(2.0, v.x);
        assert_eq!(5.0, v.y);
        assert_eq!(8.0, v.z);
        assert_eq!(3.0, v.w);
    }

    #[test]
    fn splat() {
        let v = Vec4::splat(2.0);
        assert_eq!(2.0, v.x);
        assert_eq!(2.0, v.y);
        assert_eq!(2.0, v.z);
        assert_eq!(2.0, v.w);
    }

    #[test]
    fn constants() {
        assert_eq!(Vec4::new(1.0, 1.0, 1.0, 1.0), Vec4::ONE);
        assert_eq!(Vec4::new(0.0, 0.0, 0.0, 0.0), Vec4::ZERO);
        assert_eq!(Vec4::new(1.0, 0.0, 0.0, 0.0), Vec4::X);
        assert_eq!(Vec4::new(0.0, 1.0, 0.0, 0.0), Vec4::Y);
        assert_eq!(Vec4::new(0.0, 0.0, 1.0, 0.0), Vec4::Z);
        assert_eq!(Vec4::new(0.0, 0.0, 0.0, 1.0), Vec4::W);
    }

    #[test]
    fn index() {
        let v1 = Vec4::new(1.0, 2.0, 4.0, 5.0);
        assert_eq!(1.0, v1[0]);
        assert_eq!(2.0, v1[1]);
        assert_eq!(4.0, v1[2]);
        assert_eq!(5.0, v1[3]);
    }

    #[test]
    fn index_mut() {
        let mut v1 = Vec4::new(1.0, 2.0, 4.0, 5.0);
        v1[0] = 9.0;
        v1[1] = 3.0;
        v1[2] = 6.0;
        v1[3] = 7.0;
        assert_eq!(9.0, v1[0]);
        assert_eq!(3.0, v1[1]);
        assert_eq!(6.0, v1[2]);
        assert_eq!(7.0, v1[3]);
    }

    #[test]
    fn addition() {
        let v = Vec4::new(3.0, 4.0, 6.0, 8.0);
        let w = Vec4::new(5.0, 2.0, 1.0, 7.0);
        assert_eq!(Vec4::new(8.0, 6.0, 7.0, 15.0), v + w);
    }

    #[test]
    fn subtraction() {
        let v = Vec4::new(3.0, 4.0, 6.0, 8.0);
        let w = Vec4::new(5.0, 2.0, 1.0, 7.0);
        assert_eq!(Vec4::new(-2.0, 2.0, 5.0, 1.0), v - w);
    }

    #[test]
    fn scalar_multiplication() {
        let v = Vec4::new(3.0, 4.0, 6.0, 8.0) * 2.0;
        assert_eq!(6.0, v.x);
        assert_eq!(8.0, v.y);
        assert_eq!(12.0, v.z);
        assert_eq!(16.0, v.w);
    }

    #[test]
    fn scalar_multiplication_assign() {
        let mut v = Vec4::new(3.0, 4.0, 6.0, 8.0);
        v *= 2.0;
        assert_eq!(6.0, v.x);
        assert_eq!(8.0, v.y);
        assert_eq!(12.0, v.z);
        assert_eq!(16.0, v.w);
    }

    #[test]
    fn scalar_division() {
        let v = Vec4::new(12.0, 16.0, 24.0, 32.0) / 4.0;
        assert_eq!(3.0, v.x);
        assert_eq!(4.0, v.y);
        assert_eq!(6.0, v.z);
        assert_eq!(8.0, v.w);
    }

    #[test]
    fn scalar_division_assign() {
        let mut v = Vec4::new(12.0, 16.0, 24.0, 32.0);
        v /= 4.0;
        assert_eq!(3.0, v.x);
        assert_eq!(4.0, v.y);
        assert_eq!(6.0, v.z);
        assert_eq!(8.0, v.w);
    }

    #[test]
    fn opposite() {
        let v = -Vec4::new(3.0, -4.0, 6.0, -2.0);
        assert_eq!(-3.0, v.x);
        assert_eq!(4.0, v.y);
        assert_eq!(-6.0, v.z);
        assert_eq!(2.0, v.w);
    }

    #[test]
    fn dot_product() {
        let v1 = Vec4::new(3.0, 4.0, 7.0, 1.0);
        let v2 = Vec4::new(6.0, 2.0, 5.0, 2.0);
        assert_eq!(63.0, v1.dot(v2));
    }

    #[test]
    fn magnitude() {
        let v1 = Vec4::new(3.0, 4.0, 7.0, 1.0);
        assert_eq!((9.0f32 + 16.0f32 + 49.0f32 + 1.0f32).sqrt(), v1.magnitude());
    }

    #[test]
    fn magnitude_squared() {
        let v = Vec4::new(3.0, 4.0, 6.0, 2.0);
        assert_eq!(
            3f32 * 3f32 + 4f32 * 4f32 + 6f32 * 6f32 + 2f32 * 2f32,
            v.magnitude_squared()
        );
    }

    #[test]
    fn normalize() {
        let v1 = Vec4::new(3.0, 4.0, 7.0, 2.0);
        let length = (3f32 * 3.0 + 4.0 * 4.0 + 7.0 * 7.0 + 2.0 * 2.0).sqrt();
        let normalized = v1.normalize();
        assert_eq!(3.0 / length, normalized.x);
        assert_eq!(4.0 / length, normalized.y);
        assert_eq!(7.0 / length, normalized.z);
        assert_eq!(2.0 / length, normalized.w);
    }

    #[test]
    fn as_ref() {
        let v = Vec4::new(3.0, 4.0, 6.0, 1.0);
        assert_eq!(&[3.0, 4.0, 6.0, 1.0], v.as_ref());
        assert_eq!(3.0, v.as_ref()[0]);
        assert_eq!(4.0, v.as_ref()[1]);
        assert_eq!(6.0, v.as_ref()[2]);
        assert_eq!(1.0, v.as_ref()[3]);
    }

    #[test]
    fn as_mut() {
        let mut v = Vec4::new(3.0, 4.0, 6.0, 1.0);
        let mut_ref_v = v.as_mut();
        mut_ref_v[0] = 2.0;
        assert_eq!(&[2.0, 4.0, 6.0, 1.0], v.as_mut());
        assert_eq!(2.0, v.as_ref()[0]);
        assert_eq!(4.0, v.as_ref()[1]);
        assert_eq!(6.0, v.as_ref()[2]);
        assert_eq!(1.0, v.as_ref()[3]);
    }

    #[test]
    fn from_slice() {
        let v: Vec4 = [2.0, 3.0, 4.0, 1.0].into();
        assert_eq!(2.0, v.x);
        assert_eq!(3.0, v.y);
        assert_eq!(4.0, v.z);
        assert_eq!(1.0, v.w);
    }

    #[test]
    fn as_slice() {
        let v = Vec4::new(3.0, 4.0, 6.0, 1.0);
        assert_eq!(&[3.0, 4.0, 6.0, 1.0], v.as_slice());
        assert_eq!(4, v.as_slice().len());
        assert_eq!(3.0, v.as_slice()[0]);
        assert_eq!(4.0, v.as_slice()[1]);
        assert_eq!(6.0, v.as_slice()[2]);
        assert_eq!(1.0, v.as_slice()[3]);
    }

    #[test]
    fn as_mut_slice() {
        let mut v = Vec4::new(3.0, 4.0, 6.0, 1.0);
        let mut_ref_v = v.as_mut_slice();
        assert_eq!(4, mut_ref_v.len());
        mut_ref_v[0] = 2.0;
        assert_eq!(&[2.0, 4.0, 6.0, 1.0], v.as_slice());
    }

    #[test]
    fn as_ptr() {
        let v = Vec4::new(3.0, 4.0, 6.0, 1.0);
        assert_eq!(
            std::ptr::addr_of!(v) as *const c_void,
            v.as_ptr() as *const c_void
        );
    }

    #[test]
    fn as_mut_ptr() {
        let mut v = Vec4::new(3.0, 4.0, 6.0, 1.0);
        assert_eq!(
            std::ptr::addr_of_mut!(v) as *mut c_void,
            v.as_mut_ptr() as *mut c_void
        );
    }
}
