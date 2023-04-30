use std::{
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
    ptr::{slice_from_raw_parts, slice_from_raw_parts_mut},
};

/// Vec4 is a vector of 4 float components.
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
    pub const W: Self = Self::new(0.0, 0.0, 1.0, 1.0);

    /// Creates a new vector with all components set to `v`.
    #[inline(always)]
    pub const fn splat(v: f32) -> Self {
        Vec4 {
            x: v,
            y: v,
            z: v,
            w: v,
        }
    }
    /// Creates a new vector from values.
    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Vec4 { x, y, z, w }
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
    /// Same as the dot product with itself:
    /// ```
    /// use simple_math_lib::Vec4;
    /// let v = Vec4::new(3.0, 4.0, 6.0, 2.0);
    /// assert_eq!((3 * 3 + 4 * 4 + 6 * 6 + 2 * 2) as f32, v.magnitude_squared());
    /// ```
    #[doc(alias = "length")]
    #[inline]
    pub fn magnitude_squared(self) -> f32 {
        self.dot(self)
    }

    /// Gets the magnitude (or length) of the vector.
    ///
    /// Equivalent to the square root of  the dot product with itself.
    #[doc(alias = "length")]
    #[inline]
    pub fn magnitude(self) -> f32 {
        self.magnitude_squared().sqrt()
    }

    /// Normalizes the vector means return a vector with the same direction and a magnitude of 1.
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
