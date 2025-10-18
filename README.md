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

int main() {
    Graph mgr;
    mgr.init_manager();

    Var x = mgr.variable(2.0);
    Var y = mgr.variable(3.0);
    Var z = (x + y) * sin(x);  // z = (x + y) * sin(x)

    mgr.backward(z);

    std::cout << "dz/dx = " << x.grad() << "\n";  // ≈ -1.1712
    std::cout << "dz/dy = " << y.grad() << "\n";  // ≈  0.9093
}
```
**Gradient Based Optimization**
```cpp
int main() {
    Graph mgr;
    mgr.init_manager();

    Var x = mgr.variable(0.0);  

    const double learning_rate = 0.1;
    for (int step = 0; step < 10; ++step) {
        // f(x) = (x - 3)^2
        Var loss = (x - 3.0) * (x - 3.0);

        mgr.backward(loss);


        double new_x_value = x.value() - learning_rate * x.grad();

        
        x = mgr.variable(new_x_value);

        std::cout << "Step " << step << ": x = " << x.value()
                  << ", loss = " << loss.value() << "\n";
    }
}
```
**Custom Derivative Calculation**
```cpp


int main() {
    Graph mgr;
    mgr.init_manager();

    Var x = mgr.variable(1.5);
    Var y = mgr.variable(0.8);
    Var z = mgr.variable(2.0);

    // L = exp(x*y) + log(z + 1) - sin(x)*cos(y) + (x - z)^2
    Var L = exp(x * y) + log(z + 1.0) - sin(x) * cos(y) + (x - z) * (x - z);

    mgr.backward(L);

    std::cout << "dL/dx = " << x.grad() << "\n";
    std::cout << "dL/dy = " << y.grad() << "\n";
    std::cout << "dL/dz = " << z.grad() << "\n";
}
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

## 💡 Philosophy

- **Learn by building**: No black boxes
- **CPU-first**: Optimize for available hardware
- **Interpretability over convenience**: Understanding trumps speed
- **Minimal dependencies**: Standard C++ only

---

## 🤝 Contributing

Personal educational project — feedback and collaboration welcome!

**Author**: Vladimir Ryzhkov  
**License**: MIT
