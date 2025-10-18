// src/scalar_autodiff.cpp
#include <autodiff/scalar_autodiff.h>
#include <cassert>
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
namespace scautodiff {
  size_t Graph::const_node(double value) {
    Node node;
    node.value = value;
    node.index = tape.size();
    node.op = OpType::CONST;
    tape.push_back(node);
    return node.index;
  }

  size_t Graph::input_node(double value) {
    Node node;
    node.value = value;
    node.index = tape.size();
    node.op = OpType::INPUT;
    tape.push_back(node);
    return node.index;
  }
  size_t Graph::add(const size_t a, const size_t b) {
        Node result;
        result.value = tape[a].value + tape[b].value;
        result.input0 = tape[a].index;
        result.input1 = tape[b].index;
        result.index = tape.size();
        result.op = OpType::ADD;
        tape.push_back(result);
        return result.index;
    }
    size_t Graph::sub(const size_t a, const size_t b) {
        Node result;
        result.value = tape[a].value - tape[b].value;
        result.input0 = tape[a].index;
        result.input1 = tape[b].index;
        result.index = tape.size();
        result.op = OpType::SUB;
        tape.push_back(result);
        return result.index;
    }
    size_t Graph::mul(const size_t a, const size_t b) {
        Node result;
        result.value = tape[a].value * tape[b].value;
        result.input0 = tape[a].index;
        result.input1 = tape[b].index;
        result.index = tape.size();
        result.op = OpType::MUL;
        tape.push_back(result);
        return result.index;
    }
    size_t Graph::div(const size_t a, const size_t b) {
        Node result;
        result.value = tape[a].value / tape[b].value;
        result.input0 = tape[a].index;
        result.input1 = tape[b].index;
        result.index = tape.size();
        result.op = OpType::DIV;
        tape.push_back(result);
        return result.index;
    }
    size_t Graph::power(const size_t a, const size_t b) {
        Node result;
        result.value = std::pow(tape[a].value,tape[b].value);
        result.input0 = tape[a].index;
        result.input1 = tape[b].index;
        result.index = tape.size();
        result.op = OpType::POW;
        tape.push_back(result);
        return result.index;
    }
    size_t Graph::sinus(const size_t a) {
        Node result;
        result.value = std::sin(tape[a].value);
        result.input0 = tape[a].index;
        result.input1 = 0;
        result.index = tape.size();
        result.op = OpType::SIN;
        tape.push_back(result);
        return result.index;
    }
    size_t Graph::cosine(const size_t a) {
        Node result;
        result.value = std::cos(tape[a].value);
        result.input0 = tape[a].index;
        result.input1 = 0;
        result.index = tape.size();
        result.op = OpType::COS;
        tape.push_back(result);
        return result.index;
    }
    size_t Graph::neg(const size_t a) {
        Node result;
        result.value = -tape[a].value;
        result.input0 = tape[a].index;
        result.input1 = 0;
        result.index = tape.size();
        result.op = OpType::NEG;
        tape.push_back(result);
        return result.index;
    }
    size_t Graph::square(const size_t a) {
        Node result;
        result.value = tape[a].value * tape[a].value;
        result.input0 = tape[a].index;
        result.input1 = 0;
        result.index = tape.size();
        result.op = OpType::SQUARE;
        tape.push_back(result);
        return result.index;
    }
    size_t Graph::log(const size_t a) {
        Node result;
        result.value = std::log(tape[a].value);
        result.input0 = tape[a].index;
        result.input1 = 0;
        result.index = tape.size();
        result.op = OpType::LOG;
        tape.push_back(result);
        return result.index;
    }
    size_t Graph::exp(const size_t a) {
        Node result;
        result.value = std::exp(tape[a].value);
        result.input0 = tape[a].index;
        result.input1 = 0;
        result.index = tape.size();
        result.op = OpType::EXP;
        tape.push_back(result);
        return result.index;
    }


    // В graph.cpp, в namespace scautodiff

    const std::array<Graph::MulDerivatives(*)(double, double), 13> Graph::DERIVATIVE_TABLE = {{
        // CONST
        [](double a, double b) -> Graph::MulDerivatives { return {0.0, 0.0}; },
        // INPUT
        [](double a, double b) -> Graph::MulDerivatives { return {1.0, 0.0}; },
        // ADD
        [](double a, double b) -> Graph::MulDerivatives { return {1.0, 1.0}; },
        // SUB
        [](double a, double b) -> Graph::MulDerivatives { return {1.0, -1.0}; },
        // MUL
        [](double a, double b) -> Graph::MulDerivatives { return {b, a}; },
        // DIV
        [](double a, double b) -> Graph::MulDerivatives {
            return {1.0 / b, -a / (b * b)};
        },
        // POW
        [](double a, double b) -> Graph::MulDerivatives {
            const double f = std::pow(a, b);
            return {b * std::pow(a, b - 1), f * std::log(a)};
        },
        // SIN
        [](double a, double) -> Graph::MulDerivatives { return {std::cos(a), 0.0}; },
        // COS
        [](double a, double) -> Graph::MulDerivatives { return {-std::sin(a), 0.0}; },
        // NEG
        [](double a, double) -> Graph::MulDerivatives { return {-1.0, 0.0}; },
        // SQUARE
        [](double a, double) -> Graph::MulDerivatives { return {2.0 * a, 0.0}; },
        // EXP
        [](double a, double) -> Graph::MulDerivatives { return {std::exp(a), 0.0}; },
        // LOG
        [](double a, double) -> Graph::MulDerivatives { return {1.0 / a, 0.0}; }
    }};

    [[nodiscard]] Graph::MulDerivatives Graph::derivative(const Node& node) const {
        const double operand1 = tape[node.input0].value;
        const double operand2 = tape[node.input1].value;
        auto fn = DERIVATIVE_TABLE[static_cast<size_t>(node.op)];
        return  fn(operand1, operand2);
    }
    [[nodiscard]]const Graph::Node& Graph::node_from_tape(const size_t index) const {
        return tape.at(index);
    }
    void Graph::init_graph() {
      if (tape.empty()) {
          tape.push_back(nullNode);
      }
      else
          tape[0] = nullNode;
    }
    Graph::Var Graph::variable(const double value) {
      const size_t idx = input_node(value);
      return {idx, this};
    }
    Graph::Var Graph::constant(const double value) {
      const size_t idx = const_node(value);
      return {idx, this};
    }
    void Graph::check_same_manager(const Graph::Var& a, const Graph::Var& b) {
      if (a.manager != b.manager) {
          throw std::invalid_argument("Variables must belong to the same autodiff manager");
      }
    }
    void Graph::backward(const Graph::Var& output) {
      for (auto& node : tape) {
          node.grad = 0.0;
      }
      tape[output.index].grad = 1.0;
      for (size_t i = tape.size(); i-- > 0;) {
          if (tape[i].input0 != 0) tape[tape[i].input0].grad += derivative(tape[i]).da*tape[i].grad;
          if (tape[i].input1 != 0) tape[tape[i].input1].grad += derivative(tape[i]).db*tape[i].grad;
      }
    }

    [[nodiscard]] double Graph::Var::value() const { return manager->node_from_tape(index).value;}
    [[nodiscard]] double Graph::Var::grad() const {return manager->node_from_tape(index).grad;}
    using Var = Graph::Var;


    Var operator+(const Var& a, const Var& b) {
        Graph::check_same_manager(a, b);
        return {a.manager->add(a.index, b.index), a.manager};
    }

    Var operator-(const Var& a, const Var& b) {
        Graph::check_same_manager(a, b);
        return {a.manager->sub(a.index, b.index), a.manager};
    }

    Var operator*(const Var& a, const Var& b) {
        Graph::check_same_manager(a, b);
        return {a.manager->mul(a.index, b.index), a.manager};
    }

    Var operator/(const Var& a, const Var& b) {
        Graph::check_same_manager(a, b);
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
}

