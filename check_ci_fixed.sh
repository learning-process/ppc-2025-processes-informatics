#!/bin/bash

echo "🔍 CI Check with System Tools"
echo "=============================="

# Используем системные инструменты
CLANG_FORMAT="/usr/bin/clang-format"
CLANG_TIDY="/usr/bin/clang-tidy"

# 1. Проверка форматирования
echo "1. Checking code formatting..."
if [ -f "$CLANG_FORMAT" ]; then
    find tasks/sizov_d_string_mismatch_count -name "*.cpp" -o -name "*.hpp" | \
        xargs $CLANG_FORMAT --dry-run --Werror
    if [ $? -eq 0 ]; then
        echo "✅ Formatting: OK"
    else
        echo "❌ Formatting: Issues found"
        exit 1
    fi
else
    echo "⚠️  clang-format not found, skipping formatting check"
fi

# 2. Проверка clang-tidy
echo "2. Running clang-tidy..."
mkdir -p build
cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .. > /dev/null 2>&1
cd ..

FAILED=0
for file in $(find tasks/sizov_d_string_mismatch_count -name "*.cpp" -o -name "*.hpp"); do
    echo "   $file"
    $CLANG_TIDY -p build "$file" 2>&1 | grep -q "error:" && {
        echo "   ❌ Errors found"
        $CLANG_TIDY -p build "$file" 2>&1 | grep "error:" | head -2
        FAILED=1
    } || echo "   ✅ OK"
done

if [ $FAILED -eq 0 ]; then
    echo "✅ Clang-tidy: All files passed"
else
    echo "❌ Clang-tidy: Some files failed"
    exit 1
fi

# 3. Проверка сборки
echo "3. Checking build..."
cd build
if make -j4 > /dev/null 2>&1; then
    echo "✅ Build: Successful"
else
    echo "❌ Build: Failed"
    exit 1
fi
cd ..

echo "=============================="
echo "🎉 All checks passed!"
