// include/autodiff/scalar_autodiff.h
#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <array>
namespace scautodiff {
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
    private:
        enum class OpType : size_t {
            CONST,
            INPUT,
            ADD,
            SUB,
            MUL,
            DIV,
            POW,
            SIN,
            COS,
            NEG,
            SQUARE,
            EXP,
            LOG
        };
        struct Node {
            double value = 0;
            double grad = 0;
            size_t index = 0;
            size_t input0 = 0;
            size_t input1 = 0;
            OpType op = OpType::CONST;
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
        size_t const_node(double value);
        size_t input_node(double value);
        size_t add(size_t a, size_t b);
        size_t sub(size_t a, size_t b);
        size_t mul(size_t a, size_t b);
        size_t div(size_t a, size_t b);
        size_t power(size_t a, size_t b);
        size_t sinus(size_t a);
        size_t cosine(size_t a);
        size_t neg(size_t a);
        size_t square(size_t a);
        size_t log(size_t a);
        size_t exp(size_t a);
        // MulDerivatives holds partial derivatives of a binary operation f(a, b):
        // - da = ∂f/∂a  (sensitivity w.r.t. first input)
        // - db = ∂f/∂b  (sensitivity w.r.t. second input)
        struct MulDerivatives {double da, db;};
        /*
        Analytical partial derivatives for elementary operations.
        Each function computes (∂f/∂a, ∂f/∂b) for a binary operation f(a, b).
        For unary operations, the second derivative (∂f/∂b) is always 0.
        These are used during the backward pass to propagate gradients through the computation graph.
        */
        static MulDerivatives add_derivative(double a, double b);
        static MulDerivatives mul_derivative(double a, double b);
        static MulDerivatives sub_derivative(double a, double b);
        static MulDerivatives div_derivative(double a, double b);
        static MulDerivatives pow_derivative(double a, double b);
        static MulDerivatives sin_derivative(double a, double b);
        static MulDerivatives cos_derivative(double a, double b);
        static MulDerivatives neg_derivative(double a, double b);
        static MulDerivatives square_derivative(double a, double b);
        static MulDerivatives exp_derivative(double a, double b);
        static MulDerivatives log_derivative(double a, double b);
        static MulDerivatives const_derivative(double a, double b);
        static MulDerivatives null_derivative(double a, double b);
        // Внутри namespace scautodiff, рядом с Graph


        [[nodiscard]] MulDerivatives derivative(const Node& node) const;
        [[nodiscard]] const Node& node_from_tape(size_t index) const;
        static const std::array<MulDerivatives(*)(double, double), 13> DERIVATIVE_TABLE;
    public:
        class Var {
        private:
            size_t index;
            Graph* manager;
        public:
            /*
            Var is a user-facing handle to a node in the computation graph.
            It encapsulates the internal node index and a pointer to its owning Graph,
            providing safe access to the forward value and backward gradient.
            Users interact only with Var objects—never with raw indices or tape internals.
            */
            Var(const size_t idx, Graph* mgr) : index(idx), manager(mgr) {}
            [[nodiscard]] double value() const;
            [[nodiscard]] double grad() const;
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

        // Ensures the computation graph starts with a valid null node at tape[0].
        // This node acts as a sentinel: input indices equal to 0 mean "no parent".
        // Safe to call multiple times (idempotent).
        void init_graph();
        void backward(const Var& output);
        Var variable(double value);
        Var constant(double value);
        static void check_same_manager(const Var& a, const Var& b);
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

}




