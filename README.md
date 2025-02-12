# kwanmath
Mathematics the Kwan Systems way

# kwanmath.vector
Vectors the Kwan Systems way. Includes features to make vector math as convenient as it is in POV-Ray, and is inspired by the latter's syntax.

* Dot Product (vdot): Calculates the dot product of two vectors or stacks of vectors, with support for broadcasting.
* Vector Length (vlength): Calculates the length of a vector as the square root of its dot product with itself.
* Normalization (vnormalize): Calculates a unit-length vector parallel to the given vector.
* Angle Calculation (vangle): Calculates the angle between two vectors, with options to return the result in degrees or radians.
* Cross Product (vcross): Calculates the cross-product of two vectors or stacks of vectors.
* Vector Composition and Decomposition (vcomp, vdecomp): Provides functions to compose a stack of vectors from components and decompose a vector into its components, respectively.

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

