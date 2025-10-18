#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>


class Graph {

private:

    //Node is a basic domain of our structure
    /*
        double value - value of this node then you go forward
        double grad - a summ of all children's derivatives
        int index - an index at the tape
        unsigned int input0 - index of first operand
        unsigned int input1  - index of second operand
        std::string op - operation type (exp. add is addition)
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
    //tape is a memory main data-structure, it will be used for organizing out calculation tree on it
    std::vector<Node> tape;
    //first elem of tape is nullNode, it is like nullPtr, technical unit, which means you haven't got a father
    Node nullNode;

        struct MulDerivatives {
        double da;
        double db;
    };

    size_t const_node(double value){
        Node node;
        node.value = value;
        node.index = tape.size();
        node.op = "const";
        tape.push_back(node);
        return node.index;
    }
    size_t input_node(double value){
        Node node;
        node.value = value;
        node.index = tape.size();
        node.op = "input";
        tape.push_back(node);
        return node.index;
    }
    size_t add(size_t a, size_t b) {
        Node result;
        result.value = tape[a].value + tape[b].value;
        result.input0 = tape[a].index;
        result.input1 = tape[b].index;
        result.index = tape.size();
        result.op = "add";
        tape.push_back(result);
        return result.index;
    }
    size_t sub(size_t a, size_t b) {
        Node result;
        result.value = tape[a].value - tape[b].value;
        result.input0 = tape[a].index;
        result.input1 = tape[b].index;
        result.index = tape.size();
        result.op = "sub";
        tape.push_back(result);
        return result.index;
    }
    size_t mul(size_t a, size_t b) {
        Node result;
        result.value = tape[a].value * tape[b].value;
        result.input0 = tape[a].index;
        result.input1 = tape[b].index;
        result.index = tape.size();
        result.op = "mul";
        tape.push_back(result);
        return result.index;
    }
    size_t div(size_t a, size_t b) {
        Node result;
        result.value = tape[a].value / tape[b].value;
        result.input0 = tape[a].index;
        result.input1 = tape[b].index;
        result.index = tape.size();
        result.op = "div";
        tape.push_back(result);
        return result.index;
    }
    size_t power(size_t a, size_t b) {
        Node result;
        result.value = std::pow(tape[a].value,tape[b].value);
        result.input0 = tape[a].index;
        result.input1 = tape[b].index;
        result.index = tape.size();
        result.op = "pow";
        tape.push_back(result);
        return result.index;
    }
    size_t sinus(size_t a) {
        Node result;
        result.value = std::sin(tape[a].value);
        result.input0 = tape[a].index;
        result.input1 = 0;
        result.index = tape.size();
        result.op = "sin";
        tape.push_back(result);
        return result.index;
    }
    size_t cosine(size_t a) {
        Node result;
        result.value = std::cos(tape[a].value);
        result.input0 = tape[a].index;
        result.input1 = 0;
        result.index = tape.size();
        result.op = "cos";
        tape.push_back(result);
        return result.index;
    }
    size_t neg(size_t a) {
        Node result;
        result.value = -tape[a].value;
        result.input0 = tape[a].index;
        result.input1 = 0;
        result.index = tape.size();
        result.op = "neg";
        tape.push_back(result);
        return result.index;
    }
    size_t square(size_t a) {
        Node result;
        result.value = tape[a].value * tape[a].value;
        result.input0 = tape[a].index;
        result.input1 = 0;
        result.index = tape.size();
        result.op = "square";
        tape.push_back(result);
        return result.index;
    }
    size_t log(size_t a) {
        Node result;
        result.value = std::log(tape[a].value);
        result.input0 = tape[a].index;
        result.input1 = 0;
        result.index = tape.size();
        result.op = "log";
        tape.push_back(result);
        return result.index;
    }
    size_t exp(size_t a) {
        Node result;
        result.value = std::exp(tape[a].value);
        result.input0 = tape[a].index;
        result.input1 = 0;
        result.index = tape.size();
        result.op = "exp";
        tape.push_back(result);
        return result.index;
    }

    static MulDerivatives add_derivative(double a, double b) {
        return {1.0, 1.0};
    }

    static MulDerivatives mul_derivative(double a, double b) {
        return {b, a};
    }

    static MulDerivatives sub_derivative(double a, double b) {
        return {1.0, -1.0}; // f = a - b
    }

    static MulDerivatives div_derivative(double a, double b) {
        return {1.0 / b, -a / (b * b)};
    }

    static MulDerivatives pow_derivative(double a, double b) {
        double f = std::pow(a, b);
        double da = b * std::pow(a, b - 1);
        double db = f * std::log(a);
        return {da, db};
    }

    // Унарные (db = 0)
    static MulDerivatives sin_derivative(double a, double b) {
        return {std::cos(a), 0.0};
    }

    static MulDerivatives cos_derivative(double a, double b) {
        return {-std::sin(a), 0.0};
    }

    static MulDerivatives neg_derivative(double a, double b) {
        return {-1.0, 0.0};
    }

    static MulDerivatives square_derivative(double a, double b) {
        return {2.0 * a, 0.0};
    }

    static MulDerivatives exp_derivative(double a, double b) {
        double ea = std::exp(a);
        return {ea, 0.0};
    }

    static MulDerivatives log_derivative(double a, double b) {
        return {1.0 / a, 0.0};
    }
    static MulDerivatives const_derivative(double a, double b) {
        return {0, 0.0};
    }
    static MulDerivatives null_derivative(double a, double b) {
        return {0.0, 0.0};
    }

    static MulDerivatives just_derivative(double operand1, double operand2, const std::string& type) {
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
        return tape.at(index);  // at() безопаснее [], даёт исключение при выходе за границы
    }
public:
    class Var;
    //next block for all operations whe need at scalar step of our project
    //init_manager just for adding nullNode at the start of tape, at zero-index
    void init_manager() {
        tape.push_back(nullNode);
    }
    //next step is real autodiff? Oh, fuck it will be a brain drilling exercise, let's go bro

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
    class Var {
    private:
        size_t index;
        Graph* manager;

    public:

        Var(const size_t idx, Graph* mgr) : index(idx), manager(mgr) {}

        /// Возвращает значение переменной (результат forward pass)
        [[nodiscard]] double value() const {
            return manager->node_from_tape(index).value;
        }

        /// Возвращает градиент переменной (результат backward pass)
        [[nodiscard]] double grad() const {
            return manager->node_from_tape(index).grad;
        }
        // Внутри класса Graph, в public-секцию:
        // Бинарные операторы
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
        const size_t idx = input_node(value);  // используем твой существующий метод
        return {idx, this};           // this — указатель на текущий менеджер
    }
    Var constant(const double value) {
        const size_t idx = const_node(value);  // используй твой существующий метод
        return {idx, this};
    }

    // Вспомогательная функция для проверки менеджера
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

// Бинарные операторы
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

// Унарный минус
Var operator-(const Var& a) {
    return {a.manager->neg(a.index), a.manager};
}

// Математические функции
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
// Var + double
Var operator+(const Var& a, double b) {
    return a + a.manager->constant(b);
}

// double + Var
Var operator+(double a, const Var& b) {
    return b.manager->constant(a) + b;
}

// Var - double
Var operator-(const Var& a, double b) {
    return a - a.manager->constant(b);
}

// double - Var
Var operator-(double a, const Var& b) {
    return b.manager->constant(a) - b;
}

// Var * double
Var operator*(const Var& a, double b) {
    return a * a.manager->constant(b);
}

// double * Var
Var operator*(const double a, const Var& b) {
    return b.manager->constant(a) * b;
}

// Var / double
Var operator/(const Var& a, double b) {
    return a / a.manager->constant(b);
}

// double / Var
Var operator/(double a, const Var& b) {
    return b.manager->constant(a) / b;
}



int main() {
    Graph mgr;
    mgr.init_manager();

    // Входные переменные (как веса или данные)
    Var x = mgr.variable(1.5);
    Var y = mgr.variable(0.8);
    Var z = mgr.variable(2.0);

    // Сложное скалярное "выражение", похожее на loss:
    // L = exp(x * y) + log(z + 1) - sin(x) * cos(y) + (x - z)^2
    Var L = exp(x * y) + log(z + 1.0) - sin(x) * cos(y) + (x - z) * (x - z);

    // Запускаем backward (dL/dL = 1)
    mgr.backward(L);

    // Получаем градиенты
    double dx = x.grad();
    double dy = y.grad();
    double dz = z.grad();

    std::cout << "Computed gradients:\n";
    std::cout << "dL/dx = " << dx << "\n";
    std::cout << "dL/dy = " << dy << "\n";
    std::cout << "dL/dz = " << dz << "\n";

    // --- Аналитическая проверка ---
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

    // Проверка с допуском (floating point)
    auto close = [](double a, double b, double eps = 1e-10) {
        return std::abs(a - b) < eps;
    };

    assert(close(dx, dLdx));
    assert(close(dy, dLdy));
    assert(close(dz, dLdz));

    std::cout << "\n✅ All gradients match! Autodiff is working correctly.\n";

    return 0;
}