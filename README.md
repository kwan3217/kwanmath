# kwanmath
Mathematics the Kwan Systems way

# kwanmath.bezier
The bezier.py module offers a variety of functions for working with cubic Bezier curves, including evaluation, splitting,
and flattening.

Key functionalities include:

* De Casteljau's Algorithm (deCasteljau): Recursively evaluates a Bezier curve at a given parameter value.
* Bezier Polynomial (B3): Computes the Bernstein polynomials for cubic Bezier functions.
* Bezier Curve Evaluation (bezier): Evaluates a cubic Bezier curve at a specified parameter value using the control points.
* Curve Flatness (flatness): Estimates the maximum distance between a Bezier curve and the line segment joining its endpoints.
* Curve Splitting (split): Splits a cubic Bezier curve into two separate curves at a specified parameter value.
* Curve Flattening (flatten): Approximates a cubic Bezier curve with a polyline within a given tolerance.
* Circular Arc Approximation (arc_l90): Generates control points for a cubic Bezier curve that approximates a circular arc up to 90 degrees.

# kwanmath.gaussian
Normal distributions in many dimensions

* 2D Gaussian Evaluation (twoD_Gaussian): Computes the value of a 2D 
  Gaussian function given parameters such as amplitude, center, standard
  deviations, and correlation coefficient.
* Covariance to Correlation Matrix (correlation_matrix): Converts a 
  covariance matrix to a correlation matrix for easier interpretation.
* 2D Gaussian Fitting (fit_twoD_Gaussian): Fits a 2D Gaussian surface
  to a given image and returns the fit parameters and evaluated Gaussian.
* Weighted Mean and Standard Deviation (mean_w, std_w): Computes the 
  weighted mean and population standard deviation of a dataset.
* In-Family Calculation (infamily): Determines which members of a 
  dataset are within a specified number of standard deviations from the mean.

# kwanmath.geodesy
Code that deals with oblate spheroids

* Geodetic coordinate transformations (lla2xyz, xyz2lla): Convert between geodetic coordinates (latitude, longitude, altitude) and rectangular coordinates.
* Spherical coordinate transformations (xyz2llr, llr2xyz): Convert between rectangular (XYZ) and spherical (longitude, latitude, radius) coordinates, with options for degrees or radians.
* Gravity Acceleration (aJ2, aTwoBody): Calculate gravitational acceleration due to J2 perturbation and two-body gravity.
* Ray-Sphere Intersection (ray_sphere_intersect): Determine the intersection point between a ray and a sphere.

# kwanmath.interp
Interpolation the Kwan Systems way.

Note that many of these are deprecated since I discovered a replacement for them in numpy

* Linear Interpolation (linterp): This function performs linear interpolation between two points, with an option to bound the output values if the input exceeds the specified range.
* Trapezoidal Interpolation (trap): Interpolation as a trapezoid. This 
  trapezoid function is one value before and after the trapezoid, a 
  different value inside the trapezoid, and with two linear segments
  bridging them. It's designed to smoothly turn an effect on, then  
  smoothly turn it back off.
* Smoothing (smooth): This function applies a boxcar average to a 1D
  dataset, effectively smoothing the data over a specified range.
* ~~Table Interpolation (tableterp)~~: Use `numpy.interp1d(x,xs,ys)` instead.
  If you want to bake a function, use `lambda x:numpy.interp1d(x,xs,ys)`.
  This function-like class creates an interpolating function from a set 
  of input-output pairs, with optional smoothing. It acts like a callable
  function that returns interpolated values based on the provided data.

# kwanmath.matrix
Matrices and rotation the Kwan Systems way.

Matrix operations are largely handled directly by numpy, with the `@` matrix multiplication operator and such things
as `hstack()`, `vstack()` etc. Inverse and other complicated functions are by `numpy.linalg`. Therefore most of this
module has to do with the matrix representation of rotations, IE elements of the SO(3) group.

Key functionalities include:

* Rotation Matrices (rot_x, rot_y, rot_z): Functions to generate rotation matrices for rotations around the primary axes (X, Y, Z).
* Arbitrary Axis Rotation (rot_axis): Generates a rotation matrix for a specified axis and angle.
* Euler Angle Rotations (euler_matrix): Constructs a rotation matrix from a set of Euler angles and specified axes.
* Point-Toward transformation (point_toward): Computes a matrix that aligns a vector from a body frame to a reference frame, ensuring directional consistency.
* Axis-Angle Conversions (m_to_aa, aa_to_m): Converts between rotation matrices and axis-angle representations, facilitating intuitive rotational transformations.

# kwanmath.ode
I am considering deprecating this whole module. It's either simple enough
to just embed the algorithm in the calling code, or so complicated that we
should use `scipy.integrate` instead.

* Euler Method (euler): Implements the simple and efficient Euler method for numerical integration of differential equations. This method is suitable for quick approximations and problems where high precision is not critical.
* Runge-Kutta Method (rk4): Provides a fourth-order Runge-Kutta solver, which offers higher accuracy compared to the Euler method. This method is ideal for more complex problems requiring precise integration over time.
* Fixed Step Calculations (calc_fixed_step, fixed_step): These utility functions help manage time step parameters, ensuring accurate and reliable integration even with floating-point precision limitations.
* Table Generation (table): Generates a table of values by repeatedly calling an integrator, useful for creating datasets and analyzing the behavior of dynamic systems over time.

# kwanmath.optimize
A wrapper around `scipy.optimize.minimize`, written before I learned how to
use the bounds for the BLFS method in `minimize()`. As they say on a datasheet
for an old electronic part, "Not recommended for new designs".

* Curve Fitting (curve_fit): A wrapper around scipy.optimize.curve_fit 
  that allows for easy activation or deactivation of parameters without
  modifying the underlying function. This function supports the inclusion
  of constraints on parameters, making it highly adaptable for various 
  fitting tasks.
* Bounded and Positive Constraints (bounded, positive): These classes 
  define functions with specific properties, such as monotonicity and
  bounded ranges. They are used to transform parameters during 
  optimization to ensure they remain within desired bounds or maintain
  certain characteristics, such as positivity. I learned this trick
  from a paper on the unscented Kalman filter, where their trial problem
  had a parameter which was restricted by the physics of the problem
  to be positive, but in a filter which didn't know how to constrain
  itself. The authors structured the problem so that instead of 
  estimating the parameter, they estimated its exponent. They transformed
  the parameter with a function which is smooth and invertible, and therefore
  easy to convert to and from, and subject to calculus. In this case they
  used `exp(x)`, which has an unlimited domain but a range of strictly 
  positive numbers. The positive constraint uses this technique with the modification
  that it passes through a chosen (x,y) with slope 1, so if x is a good
  initial guess, changes in x translate roughly proportional to changes in y.
  The function `bounded` is similar, except it is a function with unlimited
  domain but a range strictly between some upper and lower bound. It similarly
  has slope 1 at chosen point (x,y).

# kwanmath.vector
Vectors the Kwan Systems way. Includes features to make vector math as convenient as it is in POV-Ray, and is inspired by the latter's syntax.

* Dot Product (vdot): Calculates the dot product of two vectors or stacks of vectors, with support for broadcasting.
* Vector Length (vlength): Calculates the length of a vector as the square root of its dot product with itself.
* Normalization (vnormalize): Calculates a unit-length vector parallel to the given vector.
* Angle Calculation (vangle): Calculates the angle between two vectors, with options to return the result in degrees or radians.
* Cross Product (vcross): Calculates the cross-product of two vectors or stacks of vectors.
* Vector Composition and Decomposition (vcomp, vdecomp): Provides functions to compose a stack of vectors from components and decompose a vector into its components, respectively.

