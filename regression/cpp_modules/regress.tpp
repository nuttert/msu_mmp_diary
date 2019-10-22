#include "regress.hpp"

template <typename ValueType, typename ReturnType>
mtl::dense_vector<ReturnType> get_mtl_vector(const std::vector<ValueType> &stl_vector_const)
{
  auto &stl_vector = const_cast<std::vector<ValueType> &>(stl_vector_const);
  const auto size = stl_vector.size();
  mtl::dense_vector<ReturnType> vector(size, static_cast<ReturnType *>(stl_vector.data()));
  return vector;
}

template <typename ResultType, typename V1, typename V2>
ResultType operator*(const V1 &lhs, const V2 &rhs)
{
  const auto size1 = mtl::size(lhs);
  const auto size2 = mtl::size(rhs);
  const auto size = std::min(size1, size2);

  ResultType result(size);
  for (size_t i = 0; i < size; ++i)
    result[i] = lhs[i] * rhs[i];
  return result;
}

template <typename V>
MTLVector log(const V &vec)
{
  const auto size = mtl::size(vec);

  MTLVector result(size);
  for (size_t i = 0; i < size; ++i)
    result[i] = log(vec[i]);
  return result;
}

template <typename ResultType, typename V1, typename V2>
ResultType operator+(const V1 &lhs, const V2 &rhs)
{
  const auto size1 = mtl::size(lhs);
  const auto size2 = mtl::size(rhs);
  const auto size = std::min(size1, size2);

  ResultType result(size);
  for (size_t i = 0; i < size; ++i)
    result[i] = lhs[i] + rhs[i];
  return result;
}

template <typename ResultType=MTLVector, typename V,typename T>
ResultType add(const V &lhs, const T rhs)
{
  const auto size = mtl::size(lhs);

  ResultType result(size);
  for (size_t i = 0; i < size; ++i)
    result[i] = lhs[i] + rhs;
  return result;
}

template <typename ValueType>
mtl::dense_vector<ValueType> equal(const MTLVector &lhs, const MTLVector &rhs)
{
  const auto size1 = mtl::size(lhs);
  const auto size2 = mtl::size(rhs);
  const auto size = std::min(size1, size2);

  mtl::dense_vector<ValueType> result(size);
  for (size_t i = 0; i < size; ++i)
    result[i] = lhs[i] == rhs[i];
  return result;
}

template <typename ValueType>
mtl::dense_vector<ValueType> not_equal(const MTLVector &lhs, const MTLVector &rhs)
{
  const auto size1 = mtl::size(lhs);
  const auto size2 = mtl::size(rhs);
  const auto size = std::min(size1, size2);

  mtl::dense_vector<ValueType> result(size);
  for (size_t i = 0; i < size; ++i)
    result[i] = lhs[i] != rhs[i];
  return result;
}

template <typename T, typename IndexType>
void NearestNeighbours::sort_both(std::vector<T> &v, std::vector<IndexType> &indexies)
{
  
  iota(indexies.begin(), indexies.end(), 0);
  sort(indexies.begin(), indexies.end(),
       [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
  sort(v.begin(), v.end());
}

template <typename T, typename IndexType>
void NearestNeighbours::shrink_both(std::vector<T> &v, std::vector<IndexType> &indexies)
{
  v.resize(n_neighbours);
  indexies.resize(n_neighbours);
}
