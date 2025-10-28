#!/bin/bash

set -e

echo "🚀 Starting Local CI Simulation"
echo "================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ $2${NC}"
    else
        echo -e "${RED}✗ $2${NC}"
        exit 1
    fi
}

# 1. Check .clang-format syntax
echo "1. Validating .clang-format configuration..."
clang-format -style=file -dump-config > /dev/null
print_status $? ".clang-format syntax"

# 2. Code formatting check
echo "2. Checking code formatting..."
find tasks/sizov_d_string_mismatch_count -name "*.cpp" -o -name "*.hpp" | \
    xargs clang-format --dry-run --Werror --ferror-limit=0
print_status $? "Code formatting"

# 3. Ensure build directory exists
echo "3. Setting up build environment..."
mkdir -p build
cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .. > /dev/null 2>&1
cd ..
print_status $? "Build setup"

# 4. Clang-tidy analysis
echo "4. Running clang-tidy analysis..."
FAILED_FILES=0
for file in $(find tasks/sizov_d_string_mismatch_count -name "*.cpp" -o -name "*.hpp"); do
    echo "   Analyzing: $file"
    if clang-tidy -p build "$file" --format-style=file 2>&1 | grep -q "error:\|warning:"; then
        echo -e "   ${RED}✗ Issues found${NC}"
        clang-tidy -p build "$file" --format-style=file 2>&1 | grep "error:\|warning:" | head -3
        FAILED_FILES=$((FAILED_FILES + 1))
    else
        echo -e "   ${GREEN}✓ OK${NC}"
    fi
done

if [ $FAILED_FILES -eq 0 ]; then
    echo -e "${GREEN}✓ All files passed clang-tidy${NC}"
else
    echo -e "${RED}✗ $FAILED_FILES files have clang-tidy issues${NC}"
    exit 1
fi

# 5. Build verification
echo "5. Verifying build..."
cd build
if make -j4 > build.log 2>&1; then
    echo -e "${GREEN}✓ Build successful${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    echo "Build log:"
    tail -20 build.log
    exit 1
fi
cd ..

echo "================================="
echo -e "${GREEN}🎉 All CI checks passed!${NC}"
echo "You can safely push to your branch"
