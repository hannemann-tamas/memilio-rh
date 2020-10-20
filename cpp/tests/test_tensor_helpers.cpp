#include "epidemiology/utils/tensor_helpers.h"
#include <gtest/gtest.h>
#include <array>

TEST(TestTensor, tensor_dimension_prods)
{
    ASSERT_EQ(std::vector<size_t>({4, 12, 24, 24}), epi::tensor_dimension_prods({1, 2, 3, 4}));
    ASSERT_EQ(std::vector<size_t>({2, 4, 8, 16}), epi::tensor_dimension_prods({2, 2, 2, 2}));
}

TEST(TestTensor, flatten_index_order_two)
{
    size_t n                   = 3;
    size_t m                   = 4;
    std::array<size_t, 2> dims = {n, m};

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            auto flat_index = epi::flatten_index({i, j}, dims);
            ASSERT_LT(flat_index, n * m);
            ASSERT_EQ(m * i + j, flat_index);
        }
    }
}

TEST(TestTensor, flatten_index_order_three)
{
    std::array<size_t, 3> dims = {3, 4, 5};

    for (size_t i = 0; i < dims[0]; ++i) {
        for (size_t j = 0; j < dims[1]; ++j) {
            for (size_t k = 0; k < dims[2]; ++k) {
                auto flat_index = epi::flatten_index({i, j, k}, dims);
                ASSERT_LT(flat_index, dims[0] * dims[1] * dims[2]);
                ASSERT_EQ(dims[2] * dims[1] * i + dims[2] * j + k, flat_index);
            }
        }
    }
}

TEST(TestTensor, unravel_index_order_two)
{
    size_t n = 3;
    size_t m = 4;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            auto indices = epi::unravel_index(m * i + j, {n, m});
            ASSERT_EQ(i, indices[0]);
            ASSERT_EQ(j, indices[1]);
        }
    }
}

TEST(TestTensor, unravel_index_order_three)
{
    size_t n = 3;
    size_t m = 4;
    size_t l = 5;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            for (size_t k = 0; k < l; ++k) {
                auto indices = epi::unravel_index(l * m * i + l * j + k, {n, m, l});
                ASSERT_EQ(i, indices[0]);
                ASSERT_EQ(j, indices[1]);
                ASSERT_EQ(k, indices[2]);
            }
        }
    }
}

TEST(TestTensor, get_slice_indices_order_two)
{
    /*
     * 3x3 tensor
     *
     *    0 : (0, 0)
     *    1 : (0, 1)
     *    2 : (0, 2)
     *    3 : (1, 0)
     *    4 : (1, 1)
     *    5 : (1, 2)
     *    6 : (2, 0)
     *    7 : (2, 1)
     *    8 : (2, 2)
     */
    std::array<size_t, 2> dims = {3, 3};

    // row indices
    ASSERT_EQ(std::vector<size_t>({0, 1, 2}), epi::get_slice_indices(0, 0, dims));
    ASSERT_EQ(std::vector<size_t>({3, 4, 5}), epi::get_slice_indices(0, 1, dims));
    ASSERT_EQ(std::vector<size_t>({6, 7, 8}), epi::get_slice_indices(0, 2, dims));

    // column indices
    ASSERT_EQ(std::vector<size_t>({0, 3, 6}), epi::get_slice_indices(1, 0, dims));
    ASSERT_EQ(std::vector<size_t>({1, 4, 7}), epi::get_slice_indices(1, 1, dims));
    ASSERT_EQ(std::vector<size_t>({2, 5, 8}), epi::get_slice_indices(1, 2, dims));
}

TEST(TestTensor, get_slice_indices_order_three)
{
    /*
     * 2x2x3 tensor
     *
     *    0 : (0, 0, 0)
     *    1 : (0, 0, 1)
     *    2 : (0, 0, 2)
     *    3 : (0, 1, 0)
     *    4 : (0, 1, 1)
     *    5 : (0, 1, 2)
     *    6 : (1, 0, 0)
     *    7 : (1, 0, 1)
     *    8 : (1, 0, 2)
     *    9 : (1, 1, 0)
     *   10 : (1, 1, 1)
     *   11 : (1, 1, 2)
     */
    std::array<size_t, 3> dims = {2, 2, 3};

    // first dimension
    EXPECT_EQ(std::vector<size_t>({0, 1, 2, 3, 4, 5}), epi::get_slice_indices(0, 0, dims));
    EXPECT_EQ(std::vector<size_t>({6, 7, 8, 9, 10, 11}), epi::get_slice_indices(0, 1, dims));

    // second dimension
    EXPECT_EQ(std::vector<size_t>({0, 1, 2, 6, 7, 8}), epi::get_slice_indices(1, 0, dims));
    EXPECT_EQ(std::vector<size_t>({3, 4, 5, 9, 10, 11}), epi::get_slice_indices(1, 1, dims));

    // third dimension
    EXPECT_EQ(std::vector<size_t>({0, 3, 6, 9}), epi::get_slice_indices(2, 0, dims));
    EXPECT_EQ(std::vector<size_t>({1, 4, 7, 10}), epi::get_slice_indices(2, 1, dims));
    EXPECT_EQ(std::vector<size_t>({2, 5, 8, 11}), epi::get_slice_indices(2, 2, dims));
}
