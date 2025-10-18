#include <iostream>
#include <vector>
#include <cmath>
#include <strings.h>

class autodiffNodeManager {

    private:

    //Node is a basic domain of our structure
    /*
        double forwardPass - value of this node then you go forward
        double backwardPass - a summ of all children's derevatives
        int index - an index at the tape
        unsigned int indParent1 - index of first operand
        unsigned int indParent2  - index of second operand
        std::string typeOfOperation - operation type (exp. add is addition)
     */
    class autodiffNode {
    public:
        double forwardPass = 0;
        double backwardPass = 0;
        int index = 0;
        unsigned int indParent1 = 0;
        unsigned int indParent2 = 0;
        std::string typeOfOperation = "null";
    };
    //tape is a memory main data-structure, it will be used for organizing out calculation tree on it
    std::vector<autodiffNode> tape;
    //first elem of tape is nullNode, it is like nullPtr, technical unit, which means you haven't got a father
    autodiffNode nullNode;

    public:
    autodiffNode node_from_tape(size_t index) {
        return tape[index];
    }
    //next block for all operations whe need at scalar step of our project
    size_t const_node(double value){
        autodiffNode node;
        node.forwardPass = value;
        node.index = tape.size();
        node.typeOfOperation = "const";
        tape.push_back(node);
        return node.index;
    }
    size_t input_node(double value){
        autodiffNode node;
        node.forwardPass = value;
        node.index = tape.size();
        node.typeOfOperation = "input";
        tape.push_back(node);
        return node.index;
    }
    size_t add(size_t a, size_t b) {
        autodiffNode result;
        result.forwardPass = tape[a].forwardPass + tape[b].forwardPass;
        result.indParent1 = tape[a].index;
        result.indParent2 = tape[b].index;
        result.index = tape.size();
        result.typeOfOperation = "add";
        tape.push_back(result);
        return result.index;
    }
    size_t sub(size_t a, size_t b) {
        autodiffNode result;
        result.forwardPass = tape[a].forwardPass - tape[b].forwardPass;
        result.indParent1 = tape[a].index;
        result.indParent2 = tape[b].index;
        result.index = tape.size();
        result.typeOfOperation = "sub";
        tape.push_back(result);
        return result.index;
    }
    size_t mul(size_t a, size_t b) {
        autodiffNode result;
        result.forwardPass = tape[a].forwardPass * tape[b].forwardPass;
        result.indParent1 = tape[a].index;
        result.indParent2 = tape[b].index;
        result.index = tape.size();
        result.typeOfOperation = "mul";
        tape.push_back(result);
        return result.index;
    }
    size_t div(size_t a, size_t b) {
        autodiffNode result;
        result.forwardPass = tape[a].forwardPass / tape[b].forwardPass;
        result.indParent1 = tape[a].index;
        result.indParent2 = tape[b].index;
        result.index = tape.size();
        result.typeOfOperation = "div";
        tape.push_back(result);
        return result.index;
    }
    size_t power(size_t a, size_t b) {
        autodiffNode result;
        result.forwardPass = powf64(tape[a].forwardPass,tape[b].forwardPass);
        result.indParent1 = tape[a].index;
        result.indParent2 = tape[b].index;
        result.index = tape.size();
        result.typeOfOperation = "pow";
        tape.push_back(result);
        return result.index;
    }
    size_t sinus(size_t a) {
        autodiffNode result;
        result.forwardPass = sinf64(tape[a].forwardPass);
        result.indParent1 = tape[a].index;
        result.indParent2 = 0;
        result.index = tape.size();
        result.typeOfOperation = "sin";
        tape.push_back(result);
        return result.index;
    }
    size_t cosine(size_t a) {
        autodiffNode result;
        result.forwardPass = cosf64(tape[a].forwardPass);
        result.indParent1 = tape[a].index;
        result.indParent2 = 0;
        result.index = tape.size();
        result.typeOfOperation = "cos";
        tape.push_back(result);
        return result.index;
    }
    size_t neg(size_t a) {
        autodiffNode result;
        result.forwardPass = -tape[a].forwardPass;
        result.indParent1 = tape[a].index;
        result.indParent2 = 0;
        result.index = tape.size();
        result.typeOfOperation = "neg";
        tape.push_back(result);
        return result.index;
    }
    size_t square(size_t a) {
        autodiffNode result;
        result.forwardPass = tape[a].forwardPass * tape[a].forwardPass;
        result.indParent1 = tape[a].index;
        result.indParent2 = 0;
        result.index = tape.size();
        result.typeOfOperation = "square";
        tape.push_back(result);
        return result.index;
    }
    size_t log(size_t a) {
        autodiffNode result;
        result.forwardPass = std::log(tape[a].forwardPass);
        result.indParent1 = tape[a].index;
        result.indParent2 = 0;
        result.index = tape.size();
        result.typeOfOperation = "log";
        tape.push_back(result);
        return result.index;
    }
    size_t exp(size_t a) {
        autodiffNode result;
        result.forwardPass = std::exp(tape[a].forwardPass);
        result.indParent1 = tape[a].index;
        result.indParent2 = 0;
        result.index = tape.size();
        result.typeOfOperation = "exp";
        tape.push_back(result);
        return result.index;
    }
    //init_manager just for adding nullNode at the start of tape, at zero-index
    void init_manager() {
        tape.push_back(nullNode);
    }
    //next step is real autodiff? Ough fuck it will be a braindrilling exercize, let's go bro
    struct MulDerivatives {
        double da;
        double db;
    };

    MulDerivatives add_derivative(double a, double b) {
        return {1.0, 1.0};
    }

    MulDerivatives mul_derivative(double a, double b) {
        return {b, a};
    }

    MulDerivatives sub_derivative(double a, double b) {
        return {1.0, -1.0}; // f = a - b
    }

    MulDerivatives div_derivative(double a, double b) {
        return {1.0 / b, -a / (b * b)};
    }

    MulDerivatives pow_derivative(double a, double b) {
        double f = std::pow(a, b);
        double da = b * std::pow(a, b - 1);
        double db = f * std::log(a);
        return {da, db};
    }

    // Унарные (db = 0)
    MulDerivatives sin_derivative(double a, double b) {
        return {std::cos(a), 0.0};
    }

    MulDerivatives cos_derivative(double a, double b) {
        return {-std::sin(a), 0.0};
    }

    MulDerivatives neg_derivative(double a, double b) {
        return {-1.0, 0.0};
    }

    MulDerivatives square_derivative(double a, double b) {
        return {2.0 * a, 0.0};
    }

    MulDerivatives exp_derivative(double a, double b) {
        double ea = std::exp(a);
        return {ea, 0.0};
    }

    MulDerivatives log_derivative(double a, double b) {
        return {1.0 / a, 0.0};
    }
    MulDerivatives const_derivative(double a, double b) {
        return {0, 0.0};
    }
    MulDerivatives null_derivative(double a, double b) {
        return {0.0, 0.0};
    }

    MulDerivatives just_derivative(double operand1, double operand2, const std::string& type) {
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
            int a = 1/ 0;
        }

    }
    MulDerivatives derivative(autodiffNode& node) {
        double operand1 = tape[node.indParent1].forwardPass;
        double operand2 = tape[node.indParent2].forwardPass;
        std::string typeOfOperation = node.typeOfOperation;
        MulDerivatives result = just_derivative(operand1, operand2, typeOfOperation);
        return result;
    }
    void backward() {
        for (auto& node : tape) {
            node.backwardPass = 0.0;
        }
        tape.back().backwardPass = 1;
        for (int i = tape.size() - 1; i > 0; --i) {
            if (tape[i].indParent1 != 0) tape[tape[i].indParent1].backwardPass += derivative(tape[i]).da*tape[i].backwardPass;
            if (tape[i].indParent2 != 0) tape[tape[i].indParent2].backwardPass += derivative(tape[i]).db*tape[i].backwardPass;


        }
    }



};

int main() {
    autodiffNodeManager mgr;
    mgr.init_manager();

    auto x = mgr.input_node(2.0);
    auto y = mgr.input_node(3.0);
    auto q = mgr.add(x, y);          // q = x + y
    auto sx = mgr.sinus(x);          // sin(x)
    auto z = mgr.mul(q, sx);         // z = (x+y)*sin(x)

    mgr.backward(); // предполагаем, что z — последний

    std::cout << "dz/dx = " << mgr.node_from_tape(x).backwardPass << "\n";
    std::cout << "dz/dy = " <<  mgr.node_from_tape(y).backwardPass << "\n";

    // Аналитически:
    double x_val = 2.0, y_val = 3.0;
    double dzdx = std::sin(x_val) + (x_val + y_val) * std::cos(x_val);
    double dzdy = std::sin(x_val);
    std::cout << "Expected dz/dx = " << dzdx << "\n";
    std::cout << "Expected dz/dy = " << dzdy << "\n";
}