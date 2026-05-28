# GR Fixtures

`hartle_thorne_slow_rotation.tn` covers the 4D stationary, axisymmetric
Hartle-Thorne slow-rotation metric ansatz:

`ds^2 = -h(r) dt^2 + dr^2/f(r) + r^2(dtheta^2 + sin(theta)^2 dphi^2) - 2 omega(r) r^2 sin(theta)^2 dt dphi`.

The fixture checks that the front end can carry a symmetric `metric4` with
non-zero `g_tphi = g_phit`. Until function-valued initial data is added to the
DSL, `h(r)`, `f(r)`, and `omega(r)` are represented by scalar parameters `h`,
`f`, and `omega` sampled at the current point.
