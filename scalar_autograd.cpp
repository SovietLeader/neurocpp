#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

/*
Graph manages a dynamic computation graph for scalar reverse-mode automatic differentiation.
It records all operations (e.g., +, *, sin) in an internal tape during the forward pass
and computes gradients via backpropagation in the backward pass.

Each variable is represented by a Var object, which hides internal node indices and
provides access to its current value and accumulated gradient.

The graph supports:
  - Binary operations: +, -, *, /
  - Unary operations: sin, cos, exp, log, negation, square
  - Mixed arithmetic with scalars (e.g., x + 2.0 or 5.0 * x)
  - Multiple uses of the same variable (gradients are summed correctly)

Usage example:
  Graph mgr;
  mgr.init_graph();
  Var x = mgr.variable(2.0);
  Var y = mgr.variable(3.0);
  Var z = (x + y) * sin(x);
  mgr.backward(z);
  std::cout << "dz/dx = " << x.grad() << "\n";

Note: init_graph() must be called once before creating variables.
All Var objects used together must belong to the same Graph instance.
*/

class Graph {
    /*
     Node represents a single operation in the computation graph.
     - value: result of the forward computation
     - grad: accumulated gradient from backward pass
     - index: position in the tape (unique ID)
     - input0, input1: indices of parent nodes (0 means no parent)
     - op: operation type (e.g., "add", "mul", "sin")
     */
    class Node {
    public:
        double value = 0;
        double grad = 0;
        size_t index = 0;
        unsigned int input0 = 0;
        unsigned int input1 = 0;
        std::string op = "null";
    };
    /*
     Tape is the main data structure that stores the computation graph.
     We use a std::vector because it provides fast sequential access and
     naturally preserves the order of node creation during the forward pass.
     */
    std::vector<Node> tape;
    // nullNode is a dummy placeholder stored at tape[0].
    // It serves as a sentinel value: an input index of 0 means "no parent" (like a null pointer).
    Node nullNode;


    /*
    The following methods create new nodes during the forward pass:
    - const_node(): creates a constant (non-differentiable) scalar value
    - input_node(): creates an input variable (differentiable, e.g., model parameter or data)
    - Binary operations (e.g., add, mul, sub, div) combine two nodes
    - Unary operations (e.g., sin, exp, log, neg) transform a single node
    */
    size_t const_node(const double value){
        Node node;
        node.value = value;
        node.index = tape.size();
        node.op = "const";
        tape.push_back(node);
        return node.index;
    }
    size_t input_node(const double value){
        Node node;
        node.value = value;
        node.index = tape.size();
        node.op = "input";
        tape.push_back(node);
        return node.index;
    }
    size_t add(const size_t a, const size_t b) {
        Node result;
        result.value = tape[a].value + tape[b].value;
        result.input0 = tape[a].index;
        result.input1 = tape[b].index;
        result.index = tape.size();
        result.op = "add";
        tape.push_back(result);
        return result.index;
    }
    size_t sub(const size_t a, const size_t b) {
        Node result;
        result.value = tape[a].value - tape[b].value;
        result.input0 = tape[a].index;
        result.input1 = tape[b].index;
        result.index = tape.size();
        result.op = "sub";
        tape.push_back(result);
        return result.index;
    }
    size_t mul(const size_t a, const size_t b) {
        Node result;
        result.value = tape[a].value * tape[b].value;
        result.input0 = tape[a].index;
        result.input1 = tape[b].index;
        result.index = tape.size();
        result.op = "mul";
        tape.push_back(result);
        return result.index;
    }
    size_t div(const size_t a, const size_t b) {
        Node result;
        result.value = tape[a].value / tape[b].value;
        result.input0 = tape[a].index;
        result.input1 = tape[b].index;
        result.index = tape.size();
        result.op = "div";
        tape.push_back(result);
        return result.index;
    }
    size_t power(const size_t a, const size_t b) {
        Node result;
        result.value = std::pow(tape[a].value,tape[b].value);
        result.input0 = tape[a].index;
        result.input1 = tape[b].index;
        result.index = tape.size();
        result.op = "pow";
        tape.push_back(result);
        return result.index;
    }
    size_t sinus(const size_t a) {
        Node result;
        result.value = std::sin(tape[a].value);
        result.input0 = tape[a].index;
        result.input1 = 0;
        result.index = tape.size();
        result.op = "sin";
        tape.push_back(result);
        return result.index;
    }
    size_t cosine(const size_t a) {
        Node result;
        result.value = std::cos(tape[a].value);
        result.input0 = tape[a].index;
        result.input1 = 0;
        result.index = tape.size();
        result.op = "cos";
        tape.push_back(result);
        return result.index;
    }
    size_t neg(const size_t a) {
        Node result;
        result.value = -tape[a].value;
        result.input0 = tape[a].index;
        result.input1 = 0;
        result.index = tape.size();
        result.op = "neg";
        tape.push_back(result);
        return result.index;
    }
    size_t square(const size_t a) {
        Node result;
        result.value = tape[a].value * tape[a].value;
        result.input0 = tape[a].index;
        result.input1 = 0;
        result.index = tape.size();
        result.op = "square";
        tape.push_back(result);
        return result.index;
    }
    size_t log(const size_t a) {
        Node result;
        result.value = std::log(tape[a].value);
        result.input0 = tape[a].index;
        result.input1 = 0;
        result.index = tape.size();
        result.op = "log";
        tape.push_back(result);
        return result.index;
    }
    size_t exp(const size_t a) {
        Node result;
        result.value = std::exp(tape[a].value);
        result.input0 = tape[a].index;
        result.input1 = 0;
        result.index = tape.size();
        result.op = "exp";
        tape.push_back(result);
        return result.index;
    }
    // MulDerivatives holds partial derivatives of a binary operation f(a, b):
    // - da = ∂f/∂a  (sensitivity w.r.t. first input)
    // - db = ∂f/∂b  (sensitivity w.r.t. second input)
    struct MulDerivatives {
        double da;
        double db;
    };
    /*
    Analytical partial derivatives for elementary operations.
    Each function computes (∂f/∂a, ∂f/∂b) for a binary operation f(a, b).
    For unary operations, the second derivative (∂f/∂b) is always 0.
    These are used during the backward pass to propagate gradients through the computation graph.
    */
    static MulDerivatives add_derivative(const double a, const double b) {
        return {1.0, 1.0};
    }

    static MulDerivatives mul_derivative(const double a, const double b) {
        return {b, a};
    }

    static MulDerivatives sub_derivative(const double a, const double b) {
        return {1.0, -1.0}; // f = a - b
    }

    static MulDerivatives div_derivative(const double a, const double b) {
        return {1.0 / b, -a / (b * b)};
    }

    static MulDerivatives pow_derivative(const double a, const double b) {
        const double f = std::pow(a, b);
        const double da = b * std::pow(a, b - 1);
        const double db = f * std::log(a);
        return {da, db};
    }

    // Унарные (db = 0)
    static MulDerivatives sin_derivative(const double a, const double b) {
        return {std::cos(a), 0.0};
    }

    static MulDerivatives cos_derivative(const double a, const double b) {
        return {-std::sin(a), 0.0};
    }

    static MulDerivatives neg_derivative(const double a, const double b) {
        return {-1.0, 0.0};
    }

    static MulDerivatives square_derivative(const double a, const double b) {
        return {2.0 * a, 0.0};
    }

    static MulDerivatives exp_derivative(const double a, const double b) {
        double ea = std::exp(a);
        return {ea, 0.0};
    }

    static MulDerivatives log_derivative(const double a, const double b) {
        return {1.0 / a, 0.0};
    }
    static MulDerivatives const_derivative(const double a, const double b) {
        return {0, 0.0};
    }
    static MulDerivatives null_derivative(const double a, const double b) {
        return {0.0, 0.0};
    }

    static MulDerivatives just_derivative(const double operand1, const double operand2, const std::string& type) {
        if (type == "add") {
            return {1,1};
        }
        else if (type == "sub") {
            return {1,-1};
        }
        else if (type == "mul") {
            return mul_derivative(operand1, operand2);
        }
        else if (type == "div") {
            return div_derivative(operand1, operand2);
        }
        else if (type == "pow") {
            return pow_derivative(operand1, operand2);
        }
        else if (type == "sin") {
            return sin_derivative(operand1, operand2);
        }
        else if (type == "cos") {
            return cos_derivative(operand1, operand2);
        }
        else if (type == "log") {
            return log_derivative(operand1, operand2);
        }
        else if (type == "exp") {
            return exp_derivative(operand1, operand2);
        }
        else if (type == "const") {
            return {0,0};
        }
        else if (type == "var") {
            return {1,0};
        }
        else if (type == "neg") {
            return neg_derivative(operand1, operand2);
        }
        else if (type == "null") {
            return null_derivative(operand1, operand2);
        }
        else {
            throw std::runtime_error("Unknown type: " + type);
        }

    }
    [[nodiscard]] MulDerivatives derivative(const Node& node) const {
        const double operand1 = tape[node.input0].value;
        const double operand2 = tape[node.input1].value;
        const std::string op = node.op;
        const MulDerivatives result = just_derivative(operand1, operand2, op);
        return result;
    }
    const Node& node_from_tape(size_t index) const {
        return tape.at(index);
    }
public:
    class Var;
    // Ensures the computation graph starts with a valid null node at tape[0].
    // This node acts as a sentinel: input indices equal to 0 mean "no parent".
    // Safe to call multiple times (idempotent).
    void init_graph() {
        if (tape.size() == 0) {
            tape.push_back(nullNode);
        }
        else
            tape[0] = nullNode;
    }
    void backward(const Var& output) {
        for (auto& node : tape) {
            node.grad = 0.0;
        }
        tape[output.index].grad = 1.0;
        for (size_t i = tape.size(); i-- > 0;) {
            if (tape[i].input0 != 0) tape[tape[i].input0].grad += derivative(tape[i]).da*tape[i].grad;
            if (tape[i].input1 != 0) tape[tape[i].input1].grad += derivative(tape[i]).db*tape[i].grad;
        }
    }
    /*
    Var is a user-facing handle to a node in the computation graph.
    It encapsulates the internal node index and a pointer to its owning Graph,
    providing safe access to the forward value and backward gradient.
    Users interact only with Var objects—never with raw indices or tape internals.
    */
    class Var {
    private:
        size_t index;
        Graph* manager;

    public:
        Var(const size_t idx, Graph* mgr) : index(idx), manager(mgr) {}
        [[nodiscard]] double value() const {
            return manager->node_from_tape(index).value;
        }
        [[nodiscard]] double grad() const {
            return manager->node_from_tape(index).grad;
        }
        friend Var operator+(const Var& a, const Var& b);
        friend Var operator+(double a, const Var& b);
        friend Var operator+(const Var& a, double b);
        friend Var operator-(const Var& a, const Var& b);
        friend Var operator-(double a, const Var& b);
        friend Var operator-(const Var& a, double b);
        friend Var operator*(const Var& a, const Var& b);
        friend Var operator*(double a, const Var& b);
        friend Var operator*(const Var& a, double b);
        friend Var operator/(const Var& a, const Var& b);
        friend Var operator/(double a, const Var& b);
        friend Var operator/(const Var& a, double b);
        friend Var operator-(const Var& a);
        friend Var sin(const Var& a);
        friend Var cos(const Var& a);
        friend Var exp(const Var& a);
        friend Var log(const Var& a);
        friend Graph;
        friend void check_same_manager(const Var& a, const Var& b);
    };
    Var variable(const double value) {
        const size_t idx = input_node(value);
        return {idx, this};
    }
    Var constant(const double value) {
        const size_t idx = const_node(value);
        return {idx, this};
    }

    static void check_same_manager(const Var& a, const Var& b) {
        if (a.manager != b.manager) {
            throw std::invalid_argument("Variables must belong to the same autodiff manager");
        }
    }
    friend Var operator+(const Var& a, const Var& b);
    friend Var operator+(double a, const Var& b);
    friend Var operator+(const Var& a, double b);
    friend Var operator-(const Var& a, const Var& b);
    friend Var operator-(double a, const Var& b);
    friend Var operator-(const Var& a, double b);
    friend Var operator*(const Var& a, const Var& b);
    friend Var operator*(double a, const Var& b);
    friend Var operator*(const Var& a, double b);
    friend Var operator/(const Var& a, const Var& b);
    friend Var operator/(double a, const Var& b);
    friend Var operator/(const Var& a, double b);
    friend Var operator-(const Var& a);
    friend Var sin(const Var& a);
    friend Var cos(const Var& a);
    friend Var exp(const Var& a);
    friend Var log(const Var& a);
    friend Graph;
    friend void check_same_manager(const Var& a, const Var& b);
};


using Var = Graph::Var;


Var operator+(const Var& a, const Var& b) {
    check_same_manager(a, b);
    return {a.manager->add(a.index, b.index), a.manager};
}

Var operator-(const Var& a, const Var& b) {
    check_same_manager(a, b);
    return {a.manager->sub(a.index, b.index), a.manager};
}

Var operator*(const Var& a, const Var& b) {
    check_same_manager(a, b);
    return {a.manager->mul(a.index, b.index), a.manager};
}

Var operator/(const Var& a, const Var& b) {
    check_same_manager(a, b);
    return {a.manager->div(a.index, b.index), a.manager};
}

Var operator-(const Var& a) {
    return {a.manager->neg(a.index), a.manager};
}

Var sin(const Var& a) {
    return {a.manager->sinus(a.index), a.manager};
}

Var cos(const Var& a) {
    return {a.manager->cosine(a.index), a.manager};
}

Var exp(const Var& a) {
    return {a.manager->exp(a.index), a.manager};
}

Var log(const Var& a) {
    return {a.manager->log(a.index), a.manager};
}

Var operator+(const Var& a, double b) {
    return a + a.manager->constant(b);
}


Var operator+(double a, const Var& b) {
    return b.manager->constant(a) + b;
}

Var operator-(const Var& a, double b) {
    return a - a.manager->constant(b);
}

Var operator-(double a, const Var& b) {
    return b.manager->constant(a) - b;
}

Var operator*(const Var& a, double b) {
    return a * a.manager->constant(b);
}

Var operator*(const double a, const Var& b) {
    return b.manager->constant(a) * b;
}

Var operator/(const Var& a, double b) {
    return a / a.manager->constant(b);
}

Var operator/(double a, const Var& b) {
    return b.manager->constant(a) / b;
}



int main() {
    Graph mgr;
    mgr.init_graph();


    Var x = mgr.variable(1.5);
    Var y = mgr.variable(0.8);
    Var z = mgr.variable(2.0);

    // L = exp(x * y) + log(z + 1) - sin(x) * cos(y) + (x - z)^2
    Var L = exp(x * y) + log(z + 1.0) - sin(x) * cos(y) + (x - z) * (x - z);

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