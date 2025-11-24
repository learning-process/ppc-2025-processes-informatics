// ============================================================================
//  SizovD Global Search — FUNCTIONAL TESTS (с отладкой по GS_DEBUG_ONLY)
// ============================================================================

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>  // DEBUG
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "sizov_d_global_search/common/include/common.hpp"
#include "sizov_d_global_search/mpi/include/ops_mpi.hpp"
#include "sizov_d_global_search/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sizov_d_global_search {

namespace {

// ============================================================================
// 1. Результат brute force
// ============================================================================

struct ReferenceResult {
  double argmin = 0.0;
  double value = 0.0;
};

// ============================================================================
// 1.1. Управление отладкой через переменную окружения GS_DEBUG_ONLY
// ============================================================================
//
// GS_DEBUG_ONLY = "case_001"  → отладка только для тестов, в имени которых
//                              встречается "case_001" (case_001_seq / mpi).
// Пустая или неустановленная переменная → отладки нет вовсе.
//

inline const char *GetDebugEnvRaw() {
  return std::getenv("GS_DEBUG_ONLY");
}

bool IsDebugGloballyEnabled() {
  const char *env = GetDebugEnvRaw();
  return env && *env;
}

bool IsDebugEnabledFor(const std::string &test_name) {
  const char *env = GetDebugEnvRaw();
  if (!env || !*env) {
    return false;
  }
  std::string needle(env);
  return test_name.find(needle) != std::string::npos;
}

// ============================================================================
// 1.2. Локальный флаг отладки парсера (LEXER/AST)
// ============================================================================

thread_local bool g_parser_debug = false;

class ParserDebugGuard {
 public:
  explicit ParserDebugGuard(bool enabled) : prev_(g_parser_debug) {
    g_parser_debug = enabled;
  }

  ~ParserDebugGuard() {
    g_parser_debug = prev_;
  }

 private:
  bool prev_;
};

// ============================================================================
// 2. ЛЕКСЕР
// ============================================================================

enum class TokenType { kEnd, kNumber, kIdentifier, kPlus, kMinus, kStar, kSlash, kCaret, kLParen, kRParen };

struct Token {
  TokenType type = TokenType::kEnd;
  std::string text{};
  double number_value = 0.0;
};

class Lexer {
 public:
  Lexer(std::string expression, std::filesystem::path path)
      : expression_(std::move(expression)), path_(std::move(path)) {
    if (g_parser_debug) {
      std::cout << "[DEBUG][functional/main.cpp] LEXER: \"" << expression_ << "\"\n";
    }
  }

  Token Next() {
    SkipWhitespace();
    if (pos_ >= expression_.size()) {
      return Token{TokenType::kEnd};
    }

    char ch = expression_[pos_];

    if (std::isdigit((unsigned char)ch) || ch == '.') {
      return ParseNumber();
    }

    if (IsIdentifierStart(ch)) {
      return ParseIdentifier();
    }

    ++pos_;
    switch (ch) {
      case '+':
        return {TokenType::kPlus, "+"};
      case '-':
        return {TokenType::kMinus, "-"};
      case '*':
        return {TokenType::kStar, "*"};
      case '/':
        return {TokenType::kSlash, "/"};
      case '^':
        return {TokenType::kCaret, "^"};
      case '(':
        return {TokenType::kLParen, "("};
      case ')':
        return {TokenType::kRParen, ")"};
    }

    throw std::runtime_error("Unexpected character '" + std::string(1, ch) + "' at position " +
                             std::to_string(pos_ - 1) + " in expression " + path_.string());
  }

 private:
  void SkipWhitespace() {
    while (pos_ < expression_.size() && std::isspace((unsigned char)expression_[pos_])) {
      ++pos_;
    }
  }

  Token ParseNumber() {
    const char *begin = expression_.c_str() + pos_;
    char *end = nullptr;
    double value = std::strtod(begin, &end);
    if (begin == end) {
      throw std::runtime_error("Invalid number in " + path_.string());
    }
    pos_ = static_cast<std::size_t>(end - expression_.c_str());
    return Token{TokenType::kNumber, "", value};
  }

  Token ParseIdentifier() {
    std::size_t start = pos_;
    while (pos_ < expression_.size() && IsIdentifierChar(expression_[pos_])) {
      ++pos_;
    }
    Token token;
    token.type = TokenType::kIdentifier;
    token.text = expression_.substr(start, pos_ - start);
    return token;
  }

  static bool IsIdentifierStart(char ch) {
    return std::isalpha((unsigned char)ch) || ch == '_';
  }
  static bool IsIdentifierChar(char ch) {
    return std::isalnum((unsigned char)ch) || ch == '_';
  }

  std::string expression_;
  std::filesystem::path path_;
  std::size_t pos_ = 0;
};

// ============================================================================
// 3. AST
// ============================================================================

class Expr {
 public:
  virtual ~Expr() = default;
  virtual double Eval(double x) const = 0;
};

using ExprPtr = std::shared_ptr<Expr>;

class ConstantExpr : public Expr {
 public:
  explicit ConstantExpr(double v) : v_(v) {}
  double Eval(double) const override {
    return v_;
  }

 private:
  double v_;
};

class VariableExpr : public Expr {
 public:
  VariableExpr() = default;
  double Eval(double x) const override {
    return x;
  }
};

class UnaryExpr : public Expr {
 public:
  UnaryExpr(char op, ExprPtr child) : op_(op), child_(std::move(child)) {}
  double Eval(double x) const override {
    double v = child_->Eval(x);
    return (op_ == '-') ? -v : v;
  }

 private:
  char op_;
  ExprPtr child_;
};

class BinaryExpr : public Expr {
 public:
  BinaryExpr(char op, ExprPtr l, ExprPtr r) : op_(op), left_(std::move(l)), right_(std::move(r)) {}

  double Eval(double x) const override {
    double a = left_->Eval(x);
    double b = right_->Eval(x);
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
    }
    return std::numeric_limits<double>::quiet_NaN();
  }

 private:
  char op_;
  ExprPtr left_, right_;
};

class FunctionCallExpr : public Expr {
 public:
  using FuncPtr = double (*)(double);

  FunctionCallExpr(FuncPtr f, ExprPtr arg) : func_(f), arg_(std::move(arg)) {}

  double Eval(double x) const override {
    return func_(arg_->Eval(x));
  }

 private:
  FuncPtr func_;
  ExprPtr arg_;
};

// ============================================================================
// 4. PARSER
// ============================================================================

class Parser {
 public:
  Parser(Lexer lex, std::filesystem::path path) : lexer_(std::move(lex)), path_(std::move(path)) {
    Advance();
  }

  ExprPtr Parse() {
    if (g_parser_debug) {
      std::cout << "[DEBUG][functional/main.cpp] AST start\n";
    }

    auto expr = ParseExpression();
    Expect(TokenType::kEnd);

    if (g_parser_debug) {
      std::cout << "[DEBUG][functional/main.cpp] AST successfully built\n";
    }
    return expr;
  }

 private:
  void Advance() {
    current_ = lexer_.Next();
  }
  bool Check(TokenType t) const {
    return current_.type == t;
  }

  void Expect(TokenType t) {
    if (!Check(t)) {
      throw std::runtime_error("Unexpected token in " + path_.string());
    }
    Advance();
  }

  ExprPtr ParseExpression() {
    auto node = ParseTerm();
    while (Check(TokenType::kPlus) || Check(TokenType::kMinus)) {
      char op = (Check(TokenType::kPlus) ? '+' : '-');
      Advance();
      auto right = ParseTerm();
      node = std::make_shared<BinaryExpr>(op, node, right);
    }
    return node;
  }

  ExprPtr ParseTerm() {
    auto node = ParseFactor();
    while (Check(TokenType::kStar) || Check(TokenType::kSlash)) {
      char op = Check(TokenType::kStar) ? '*' : '/';
      Advance();
      auto right = ParseFactor();
      node = std::make_shared<BinaryExpr>(op, node, right);
    }
    return node;
  }

  ExprPtr ParseFactor() {
    auto node = ParseUnary();
    while (Check(TokenType::kCaret)) {
      Advance();
      auto r = ParseUnary();
      node = std::make_shared<BinaryExpr>('^', node, r);
    }
    return node;
  }

  ExprPtr ParseUnary() {
    if (Check(TokenType::kPlus)) {
      Advance();
      return ParseUnary();
    }
    if (Check(TokenType::kMinus)) {
      Advance();
      return std::make_shared<UnaryExpr>('-', ParseUnary());
    }
    return ParsePrimary();
  }

  ExprPtr ParsePrimary() {
    if (Check(TokenType::kNumber)) {
      double v = current_.number_value;
      Advance();
      return std::make_shared<ConstantExpr>(v);
    }

    if (Check(TokenType::kIdentifier)) {
      std::string id = current_.text;
      Advance();

      if (Check(TokenType::kLParen)) {
        Advance();
        auto arg = ParseExpression();
        Expect(TokenType::kRParen);

        auto func = ResolveFunction(id);
        return std::make_shared<FunctionCallExpr>(func, arg);
      }

      if (id == "x") {
        return std::make_shared<VariableExpr>();
      }
      if (id == "pi") {
        return std::make_shared<ConstantExpr>(std::acos(-1));
      }

      throw std::runtime_error("Unknown identifier '" + id + "'");
    }

    if (Check(TokenType::kLParen)) {
      Advance();
      auto node = ParseExpression();
      Expect(TokenType::kRParen);
      return node;
    }

    throw std::runtime_error("Syntax error in " + path_.string());
  }

  static FunctionCallExpr::FuncPtr ResolveFunction(const std::string &name) {
    static const std::unordered_map<std::string, double (*)(double)> map = {
        {"sin", std::sin}, {"cos", std::cos},   {"tan", std::tan}, {"exp", std::exp},
        {"log", std::log}, {"sqrt", std::sqrt}, {"abs", std::fabs}};
    auto it = map.find(name);
    if (it == map.end()) {
      throw std::runtime_error("Unknown function '" + name + "'");
    }
    return it->second;
  }

  Lexer lexer_;
  Token current_;
  std::filesystem::path path_;
};

// ============================================================================
// 5. BuildFunction
// ============================================================================

Function BuildFunction(const nlohmann::json &json, const std::filesystem::path &path, bool debug) {
  if (!json.contains("expression")) {
    throw std::runtime_error("Missing expression field");
  }

  std::string expr = json.at("expression").get<std::string>();
  if (debug) {
    std::cout << "[DEBUG][functional/main.cpp] Building function: \"" << expr << "\"\n";
  }

  ParserDebugGuard guard(debug);

  Lexer lex(expr, path);
  Parser parser(std::move(lex), path);
  ExprPtr ast = parser.Parse();

  return Function([ast](double x) { return ast->Eval(x); });
}

// ============================================================================
// 6. Brute force
// ============================================================================

ReferenceResult BruteForceReference(const Problem &p, double step, bool debug) {
  if (debug) {
    std::cout << "[DEBUG][functional/main.cpp] BruteForce: " << "left=" << p.left << " right=" << p.right
              << " step=" << step << "\n";
  }

  double span = p.right - p.left;
  double safe_step = (step > 0.0 && step < span) ? step : std::max(span / 1000.0, 1e-4);

  ReferenceResult r{p.left, p.func(p.left)};

  double x = p.left;
  while (x <= p.right + 1e-12) {
    double v = p.func(x);
    if (v < r.value) {
      r.value = v;
      r.argmin = x;
    }
    x += safe_step;
  }

  double vr = p.func(p.right);
  if (vr < r.value) {
    r.value = vr;
    r.argmin = p.right;
  }

  if (debug) {
    std::cout << "[DEBUG][functional/main.cpp] Brute: x=" << r.argmin << " f=" << r.value << "\n";
  }

  return r;
}

// ============================================================================
// 7. Load tests
// ============================================================================

std::vector<TestType> LoadTestCasesFromData() {
  namespace fs = std::filesystem;
  fs::path json_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_sizov_d_global_search, "tests.json");

  if (IsDebugGloballyEnabled()) {
    std::cout << "[DEBUG][functional/main.cpp] Loading tests.json: " << json_path << "\n";
  }

  std::ifstream file(json_path);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open tests.json");
  }

  nlohmann::json data;
  file >> data;

  if (!data.is_array()) {
    throw std::runtime_error("tests.json must contain array");
  }

  std::vector<TestType> cases;
  cases.reserve(data.size());

  for (auto &item : data) {
    TestType t;
    t.name = item.at("name").get<std::string>();

    const auto &pj = item.at("problem");
    Problem p{};
    p.left = pj.at("left").get<double>();
    p.right = pj.at("right").get<double>();
    p.accuracy = pj.value("accuracy", 1e-3);
    p.reliability = pj.value("reliability", 2.0);
    p.max_iterations = pj.at("max_iterations").get<int>();

    bool debug = IsDebugEnabledFor(t.name);

    p.func = BuildFunction(item.at("function"), json_path, debug);

    if (debug) {
      std::cout << "[DEBUG][functional/main.cpp] Loaded test \"" << t.name << "\"\n";
    }

    t.problem = p;
    t.brute_force_step = item.value("brute_force_step", p.accuracy);
    t.tolerance = item.value("tolerance", p.accuracy * 5);

    cases.push_back(std::move(t));
  }

  return cases;
}

using FuncParam = ppc::util::FuncTestParam<InType, OutType, TestType>;

// ============================================================================
// 8. BuildTestTasks
// ============================================================================

std::vector<FuncParam> BuildTestTasks(const std::vector<TestType> &tests) {
  std::vector<FuncParam> tasks;
  tasks.reserve(tests.size() * 2);

  std::string mpi_name =
      std::string(ppc::util::GetNamespace<SizovDGlobalSearchMPI>()) + "_" +
      ppc::task::GetStringTaskType(SizovDGlobalSearchMPI::GetStaticTypeOfTask(), PPC_SETTINGS_sizov_d_global_search);

  std::string seq_name =
      std::string(ppc::util::GetNamespace<SizovDGlobalSearchSEQ>()) + "_" +
      ppc::task::GetStringTaskType(SizovDGlobalSearchSEQ::GetStaticTypeOfTask(), PPC_SETTINGS_sizov_d_global_search);

  for (auto &t : tests) {
    tasks.emplace_back(ppc::task::TaskGetter<SizovDGlobalSearchMPI, InType>, mpi_name, t);
    tasks.emplace_back(ppc::task::TaskGetter<SizovDGlobalSearchSEQ, InType>, seq_name, t);
  }

  return tasks;
}

}  // namespace

// ============================================================================
// 9. GTEST
// ============================================================================

class SizovDGlobalSearchFunctionalTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(
      const testing::TestParamInfo<std::tuple<std::function<std::shared_ptr<ppc::task::Task<InType, OutType>>(InType)>,
                                              std::string, TestType>> &info) {
    const TestType &test_case = std::get<2>(info.param);
    const std::string &impl = std::get<1>(info.param);

    std::string suffix = (impl.find("mpi") != std::string::npos) ? "_mpi" : "_seq";

    return test_case.name + suffix;
  }

 protected:
  void SetUp() override {
    test_case_ = std::get<(size_t)ppc::util::GTestParamIndex::kTestParams>(GetParam());
    input_ = test_case_.problem;

    debug_enabled_ = IsDebugEnabledFor(test_case_.name);

    if (debug_enabled_) {
      std::cout << "[DEBUG][functional/main.cpp] TEST START: " << test_case_.name << "\n";
    }

    reference_ = BruteForceReference(input_, test_case_.brute_force_step, debug_enabled_);
  }

  bool CheckTestOutputData(OutType &o) final {
    if (!o.converged) {
      if (debug_enabled_) {
        std::cout << "[DEBUG][functional/main.cpp] FAIL: converged=false\n";
      }
      return false;
    }

    double dx = std::abs(o.argmin - reference_.argmin);
    double dv = std::abs(o.value - reference_.value);

    if (debug_enabled_) {
      std::cout << "[DEBUG][BF] reference: xmin=" << reference_.argmin << "  f=" << reference_.value << "\n";

      std::cout << "[DEBUG][GS] result:    xmin=" << o.argmin << "  f=" << o.value << "\n";

      std::cout << "[DEBUG][GS] errors:    dx=" << dx << "  dv=" << dv << "  tol=" << test_case_.tolerance << "\n";
    }

    return dx <= test_case_.tolerance && dv <= 5 * test_case_.tolerance;
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  TestType test_case_;
  InType input_;
  ReferenceResult reference_;
  bool debug_enabled_ = false;
};

namespace {

TEST_P(SizovDGlobalSearchFunctionalTests, FindsGlobalMinimum) {
  ExecuteTest(GetParam());
}

const std::vector<TestType> kTests = LoadTestCasesFromData();
const std::vector<FuncParam> kTasks = BuildTestTasks(kTests);

INSTANTIATE_TEST_SUITE_P(GlobalSearch, SizovDGlobalSearchFunctionalTests, ::testing::ValuesIn(kTasks),
                         SizovDGlobalSearchFunctionalTests::PrintTestParam);

}  // namespace

}  // namespace sizov_d_global_search
