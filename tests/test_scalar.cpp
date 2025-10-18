    // tests/test_scalar.cpp
#include "autodiff/scalar_autodiff.h"
#include <iostream>
#include <cassert>


using namespace scautodiff;

int main() {
    Graph mgr;
    mgr.init_graph();


    Graph::Var x = mgr.variable(1.5);
    Graph::Var y = mgr.variable(0.8);
    Graph::Var z = mgr.variable(2.0);

    // L = exp(x * y) + log(z + 1) - sin(x) * cos(y) + (x - z)^2
    Graph::Var L = exp(x * y) + log(z + 1.0) - sin(x) * cos(y) + (x - z) * (x - z);

    //  (dL/dL = 1)
    mgr.backward(L);

    double dx = x.grad();
    double dy = y.grad();
    double dz = z.grad();

    std::cout << "Computed gradients:\n";
    std::cout << "dL/dx = " << dx << "\n";
    std::cout << "dL/dy = " << dy << "\n";
    std::cout << "dL/dz = " << dz << "\n";

    double xv = 1.5, yv = 0.8, zv = 2.0;

    // L = exp(x*y) + log(z+1) - sin(x)*cos(y) + (x - z)^2

    // ∂L/∂x = y*exp(x*y) - cos(x)*cos(y) + 2*(x - z)
    double dLdx = yv * std::exp(xv * yv) - std::cos(xv) * std::cos(yv) + 2.0 * (xv - zv);

    // ∂L/∂y = x*exp(x*y) + sin(x)*sin(y)
    double dLdy = xv * std::exp(xv * yv) + std::sin(xv) * std::sin(yv);

    // ∂L/∂z = 1/(z+1) - 2*(x - z)
    double dLdz = 1.0 / (zv + 1.0) - 2.0 * (xv - zv);

    std::cout << "\nAnalytical gradients:\n";
    std::cout << "dL/dx = " << dLdx << "\n";
    std::cout << "dL/dy = " << dLdy << "\n";
    std::cout << "dL/dz = " << dLdz << "\n";


    auto close = [](double a, double b, double eps = 1e-10) {
        return std::abs(a - b) < eps;
    };

    assert(close(dx, dLdx));
    assert(close(dy, dLdy));
    assert(close(dz, dLdz));

    std::cout << "\n✅ All gradients match! Autodiff is working correctly.\n";

    return 0;
}