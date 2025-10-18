cmake_minimum_required(VERSION 3.14)
project(RedSprite LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Требуем C++20 без расширений
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Путь к заголовкам (для всего проекта)
add_library(redsprite_core STATIC
    src/scalar_autodiff.cpp
)

target_include_directories(redsprite_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# --- Тест: test_scalar ---
add_executable(test_scalar
    tests/test_scalar.cpp
)
target_link_libraries(test_scalar PRIVATE redsprite_core)



# Опционально: включить предупреждения
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(redsprite_core PRIVATE -Wall -Wextra -Wpedantic)
endif()