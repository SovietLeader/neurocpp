# NeuroCPP — Modular, Interpretable Neural Networks in C++

**From scratch. With understanding. For CPU first, GPU later.**

A modern C++ deep learning framework focused on transparency, modularity, and interpretability. Built as a learning vehicle to understand neural networks from the ground up.

---

## ✅ Current Status

**Scalar Autodiff Core Complete** featuring:

- Tape-based computation graph
- Full binary and unary operations support
- Analytically verified gradients
- Clean computation/differentiation separation

---

## 🚀 Quick Examples
**Basic Scalar Autodiff**  
*Build computation graphs intuitively with operator overloading (work-in-progress API):*
```cpp
#include "neurocpp/autograd.hpp"

int main() {
    // Input variables (under active refactoring to Var API)
    auto x = input(2.0);  // Current: mgr.input_node(2.0)
    auto y = input(3.0);
    
    // Expressions with natural syntax
    auto z = (x + y) * sin(x);  // Tape records: add → sin → mul
    
    backward(z);  // Propagate gradients backward
    
    std::cout << "dz/dx = " << grad(x) << "\n";  // ≈ -1.1712
    std::cout << "dz/dy = " << grad(y) << "\n";  // ≈ 0.9093
}
```
**Gradient Based Optimization**
```cpp
// Minimize f(x) = (x - 3)^2 using gradients
auto x = input(0.0);  // Start at x=0
for (int i = 0; i < 10; ++i) {
    auto loss = square(x - 3.0);  // (x-3)^2
    backward(loss);
    
    // Update: x = x - η * ∂loss/∂x
    x.value() -= 0.1 * grad(x);  // η=0.1 (learning rate)
    
    std::cout << "x = " << x.value() << " | loss = " << loss.value() << "\n";
}
/* Output:
x = 0.6 | loss = 5.76
x = 1.08 | loss = 3.6864
... converges to x=3.0 */
```
**Custom Derivative Calculation**
```cpp
// Test: ∂/∂x [x² * sin(x)] at x=1.0
auto x = input(1.0);
auto z = square(x) * sin(x);

backward(z);
double numerical = grad(x);  // NeuroCPP's result

// Analytical derivative: 2x·sin(x) + x²·cos(x)
double analytical = 2*1.0*sin(1.0) + 1.0*1.0*cos(1.0);

assert(std::abs(numerical - analytical) < 1e-6);  // ✅ Passes!
```
---

## 🏃 Roadmap

| Phase | Status | Timeline |
|-------|--------|----------|
| **Scalar Autodiff** | ✅ Complete | Core stable |
| **Tensor Engine** | 🔄 In Progress | Next major milestone |
| **Neural Layers** | ⏳ Planned | 2025 |
| **Recurrent & Attention** | ⏳ Planned | 2025 |
| **Visualization Tools** | ⏳ Planned | 2025 |
| **CPU Optimizations** | ⏳ Planned | 2026 |
| **OpenGL Backend** | ⏳ Planned | 2026 |

---

## 🎯 Key Features

- **Transparent**: Every operation inspectable, every gradient traceable
- **Modular**: Build any architecture by wiring simple layers
- **Interpretable**: Heatmaps for activation and gradient flow
- **Portable**: Future GPU acceleration via OpenGL

---

## 🛠 Building

*Basic compilation example:*
ВСТАВЬТЕ ПРИМЕР КОМПИЛЯЦИИ ЗДЕСЬ

---

## 💡 Philosophy

- **Learn by building**: No black boxes
- **CPU-first**: Optimize for available hardware
- **Interpretability over convenience**: Understanding trumps speed
- **Minimal dependencies**: Standard C++ only

---

## 🤝 Contributing

Personal educational project — feedback and collaboration welcome!

**Author**: Vladimir Ryzhkov  
**License**: MIT (planned)
