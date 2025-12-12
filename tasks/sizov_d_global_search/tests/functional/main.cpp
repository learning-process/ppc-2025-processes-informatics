#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sizov_d_global_search/common/include/common.hpp"
#include "sizov_d_global_search/mpi/include/ops_mpi.hpp"
#include "sizov_d_global_search/seq/include/ops_seq.hpp"
#include "task/include/task.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sizov_d_global_search {

struct ExpectedSolution {
  std::vector<double> argmins{};
  double value = 0.0;
};

struct LocalTestCase {
  std::string name{};
  Problem problem{};
  ExpectedSolution expected{};
};

using FuncParam = ppc::util::FuncTestParam<InType, OutType, LocalTestCase>;

namespace expr_detail {

enum class TokenType : std::uint8_t { kEnd, kNumber, kIdentifier, kPlus, kMinus, kMul, kDiv, kPow, kLParen, kRParen };

struct Token {
  TokenType type{TokenType::kEnd};
  double number{0.0};
  std::string text{};
};

class Lexer {
 public:
  explicit Lexer(std::string s) : s_(std::move(s)) {}

  Token Next() {
    SkipSpaces();
    if (pos_ >= s_.size()) {
      return {};
    }

    const char c = s_[pos_];

    if (std::isdigit(static_cast<unsigned char>(c)) != 0 || c == '.') {
      return ParseNumber();
    }

    if (std::isalpha(static_cast<unsigned char>(c)) != 0) {
      return ParseIdentifier();
    }

    ++pos_;
    switch (c) {
      case '+':
        return {TokenType::kPlus, 0.0, "+"};
      case '-':
        return {TokenType::kMinus, 0.0, "-"};
      case '*':
        return {TokenType::kMul, 0.0, "*"};
      case '/':
        return {TokenType::kDiv, 0.0, "/"};
      case '^':
        return {TokenType::kPow, 0.0, "^"};
      case '(':
        return {TokenType::kLParen, 0.0, "("};
      case ')':
        return {TokenType::kRParen, 0.0, ")"};
      default:
        break;
    }

    throw std::runtime_error("Unexpected character");
  }

 private:
  void SkipSpaces() {
    while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_])) != 0) {
      ++pos_;
    }
  }

  Token ParseNumber() {
    const char *begin = s_.c_str() + pos_;
    char *end = nullptr;
    const double v = std::strtod(begin, &end);
    if (begin == end) {
      throw std::runtime_error("Invalid number");
    }
    pos_ = static_cast<std::size_t>(end - s_.c_str());
    return {TokenType::kNumber, v, ""};
  }

  Token ParseIdentifier() {
    const std::size_t start = pos_;
    while (pos_ < s_.size() && std::isalnum(static_cast<unsigned char>(s_[pos_])) != 0) {
      ++pos_;
    }
    return {TokenType::kIdentifier, 0.0, s_.substr(start, pos_ - start)};
  }

  std::string s_;
  std::size_t pos_{0};
};

class Expr {
 public:
  virtual ~Expr() = default;
  [[nodiscard]] virtual double Eval(double x) const = 0;
};

using ExprPtr = std::unique_ptr<Expr>;

class ConstantExpr final : public Expr {
 public:
  explicit ConstantExpr(double v) : v_(v) {}
  [[nodiscard]] double Eval(double) const override {
    return v_;
  }

 private:
  double v_;
};

class VariableExpr final : public Expr {
 public:
  [[nodiscard]] double Eval(double x) const override {
    return x;
  }
};

class UnaryExpr final : public Expr {
 public:
  UnaryExpr(char op, ExprPtr arg) : op_(op), arg_(std::move(arg)) {}

  [[nodiscard]] double Eval(double x) const override {
    const double v = arg_->Eval(x);
    return (op_ == '-') ? -v : v;
  }

 private:
  char op_;
  ExprPtr arg_;
};

class BinaryExpr final : public Expr {
 public:
  BinaryExpr(char op, ExprPtr lhs, ExprPtr rhs) : op_(op), lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

  [[nodiscard]] double Eval(double x) const override {
    const double a = lhs_->Eval(x);
    const double b = rhs_->Eval(x);
    switch (op_) {
      case '+':
        return a + b;
      case '-':
        return a - b;
      case '*':
        return a * b;
      case '/':
        return a / b;
      case '^':
        return std::pow(a, b);
      default:
        break;
    }
    return std::numeric_limits<double>::quiet_NaN();
  }

 private:
  char op_;
  ExprPtr lhs_;
  ExprPtr rhs_;
};

class FunctionCallExpr final : public Expr {
 public:
  using FuncPtr = double (*)(double);

  FunctionCallExpr(FuncPtr f, ExprPtr arg) : f_(f), arg_(std::move(arg)) {}

  [[nodiscard]] double Eval(double x) const override {
    return f_(arg_->Eval(x));
  }

 private:
  FuncPtr f_;
  ExprPtr arg_;
};

class Parser {
 public:
  explicit Parser(Lexer lex) : lex_(std::move(lex)) {
    cur_ = lex_.Next();
  }

  ExprPtr Parse() {
    auto e = ParseExpr();
    if (cur_.type != TokenType::kEnd) {
      throw std::runtime_error("Unexpected tokens after expression");
    }
    return e;
  }

 private:
  ExprPtr ParseExpr() {
    auto lhs = ParseTerm();
    while (cur_.type == TokenType::kPlus || cur_.type == TokenType::kMinus) {
      const char op = (cur_.type == TokenType::kPlus) ? '+' : '-';
      Advance();
      lhs = std::make_unique<BinaryExpr>(op, std::move(lhs), ParseTerm());
    }
    return lhs;
  }

  ExprPtr ParseTerm() {
    auto lhs = ParsePow();
    while (cur_.type == TokenType::kMul || cur_.type == TokenType::kDiv) {
      const char op = (cur_.type == TokenType::kMul) ? '*' : '/';
      Advance();
      lhs = std::make_unique<BinaryExpr>(op, std::move(lhs), ParsePow());
    }
    return lhs;
  }

  ExprPtr ParsePow() {
    auto lhs = ParseFactor();
    if (cur_.type == TokenType::kPow) {
      Advance();
      lhs = std::make_unique<BinaryExpr>('^', std::move(lhs), ParsePow());
    }
    return lhs;
  }

  ExprPtr ParseFactor() {
    if (cur_.type == TokenType::kMinus) {
      Advance();
      return std::make_unique<UnaryExpr>('-', ParseFactor());
    }

    if (cur_.type == TokenType::kNumber) {
      const double v = cur_.number;
      Advance();
      return std::make_unique<ConstantExpr>(v);
    }

    if (cur_.type == TokenType::kIdentifier) {
      const std::string id = cur_.text;
      Advance();

      if (cur_.type == TokenType::kLParen) {
        Advance();
        auto arg = ParseExpr();
        Expect(TokenType::kRParen);

        using F = double (*)(double);
        if (id == "sin") {
          return std::make_unique<FunctionCallExpr>(static_cast<F>(std::sin), std::move(arg));
        }
        if (id == "cos") {
          return std::make_unique<FunctionCallExpr>(static_cast<F>(std::cos), std::move(arg));
        }
        if (id == "exp") {
          return std::make_unique<FunctionCallExpr>(static_cast<F>(std::exp), std::move(arg));
        }
        if (id == "log") {
          return std::make_unique<FunctionCallExpr>(static_cast<F>(std::log), std::move(arg));
        }
        if (id == "sqrt") {
          return std::make_unique<FunctionCallExpr>(static_cast<F>(std::sqrt), std::move(arg));
        }
        if (id == "abs") {
          return std::make_unique<FunctionCallExpr>(static_cast<F>(std::fabs), std::move(arg));
        }

        throw std::runtime_error("Unknown function: " + id);
      }

      if (id == "x") {
        return std::make_unique<VariableExpr>();
      }
      if (id == "pi") {
        return std::make_unique<ConstantExpr>(std::acos(-1.0));
      }

      throw std::runtime_error("Unknown identifier: " + id);
    }

    if (cur_.type == TokenType::kLParen) {
      Advance();
      auto e = ParseExpr();
      Expect(TokenType::kRParen);
      return e;
    }

    throw std::runtime_error("Invalid expression");
  }

  void Expect(TokenType t) {
    if (cur_.type != t) {
      throw std::runtime_error("Unexpected token");
    }
    Advance();
  }

  void Advance() {
    cur_ = lex_.Next();
  }

  Lexer lex_;
  Token cur_{};
};

}  // namespace expr_detail

static Function BuildFunction(const nlohmann::json &json) {
  const std::string expr = json.at("expression").get<std::string>();
  expr_detail::Parser parser{expr_detail::Lexer{expr}};
  auto ast = parser.Parse();
  auto shared = std::shared_ptr<expr_detail::Expr>(std::move(ast));
  return Function{[shared](double x) { return shared->Eval(x); }};
}

static std::vector<LocalTestCase> LoadTestCasesFromData() {
  const auto path = ppc::util::GetAbsoluteTaskPath(PPC_ID_sizov_d_global_search, "tests.json");
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Cannot open tests.json");
  }

  nlohmann::json data;
  file >> data;
  if (!data.is_array()) {
    throw std::runtime_error("tests.json must be an array");
  }

  std::vector<LocalTestCase> cases;
  cases.reserve(data.size());

  for (const auto &item : data) {
    LocalTestCase tc{};
    tc.name = item.at("name").get<std::string>();

    const auto &pj = item.at("problem");
    tc.problem.left = pj.at("left");
    tc.problem.right = pj.at("right");
    tc.problem.accuracy = 1e-4;
    tc.problem.reliability = 3.0;
    tc.problem.max_iterations = 300;
    tc.problem.func = BuildFunction(item.at("function"));

    const auto &ej = item.at("expected");
    tc.expected.argmins = ej.at("argmins").get<std::vector<double>>();
    tc.expected.value = ej.at("value");

    cases.push_back(std::move(tc));
  }

  return cases;
}

static std::vector<FuncParam> BuildTestTasks(const std::vector<LocalTestCase> &tests) {
  std::vector<FuncParam> tasks;
  tasks.reserve(tests.size() * 2U);

  const std::string mpi_name =
      std::string(ppc::util::GetNamespace<SizovDGlobalSearchMPI>()) + "_" +
      ppc::task::GetStringTaskType(SizovDGlobalSearchMPI::GetStaticTypeOfTask(), PPC_SETTINGS_sizov_d_global_search);

  const std::string seq_name =
      std::string(ppc::util::GetNamespace<SizovDGlobalSearchSEQ>()) + "_" +
      ppc::task::GetStringTaskType(SizovDGlobalSearchSEQ::GetStaticTypeOfTask(), PPC_SETTINGS_sizov_d_global_search);

  for (const auto &t : tests) {
    tasks.emplace_back(ppc::task::TaskGetter<SizovDGlobalSearchMPI, InType>, mpi_name, t);
    tasks.emplace_back(ppc::task::TaskGetter<SizovDGlobalSearchSEQ, InType>, seq_name, t);
  }

  return tasks;
}

class SizovDRunFuncTestsGlobalSearch : public ppc::util::BaseRunFuncTests<InType, OutType, LocalTestCase> {
 public:
  static std::string PrintTestParam(const LocalTestCase &tc) {
    return tc.name;
  }

  void PrepareTestCase(const FuncParam &param) {
    test_case_ = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(param);
    input_ = test_case_.problem;
    expected_ = test_case_.expected;
  }

 protected:
  InType GetTestInputData() override final {
    return input_;
  }

  bool CheckTestOutputData(OutType &out) override final {
    if (!std::isfinite(out.value)) {
      return false;
    }

    const double dv = std::abs(out.value - expected_.value);
    if (dv > 20.0 * input_.accuracy) {
      return false;
    }

    if (expected_.argmins.empty()) {
      return true;
    }

    double min_dx = std::numeric_limits<double>::infinity();
    for (double a : expected_.argmins) {
      min_dx = std::min(min_dx, std::abs(out.argmin - a));
    }

    return min_dx <= 5.0 * input_.accuracy;
  }

 private:
  LocalTestCase test_case_{};
  InType input_{};
  ExpectedSolution expected_{};
};

TEST_P(SizovDRunFuncTestsGlobalSearch, FromJson) {
  PrepareTestCase(GetParam());
  ExecuteTest(GetParam());
}

namespace {

const auto kCases = LoadTestCasesFromData();
const auto kTasks = BuildTestTasks(kCases);

INSTANTIATE_TEST_SUITE_P(SizovDGlobalSearch, SizovDRunFuncTestsGlobalSearch, ::testing::ValuesIn(kTasks),
                         SizovDRunFuncTestsGlobalSearch::PrintFuncTestName<SizovDRunFuncTestsGlobalSearch>);

}  // namespace

}  // namespace sizov_d_global_search
