#pragma once
#ifndef __INCLUDE_LIBS
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/thread/thread_pool.hpp>
#include <iostream>
#include <map>
#include <math.h>
#include <memory>
#include <vector>
#include <boost/asio.hpp>
#endif

using ProbabilityMapMatrix = std::map<int, double>;
using ProbabilityVectorMatrix = std::vector<std::pair<int, double>>;
using ProbabilitiesMapTensor = std::vector<ProbabilityMapMatrix>;
using ProbabilitiesVectorTensor = std::vector<ProbabilityVectorMatrix>;

using ThreadPool = boost::asio::thread_pool;
using ThreadPoolPtr = std::shared_ptr<ThreadPool>;

using ValueType = double;
using Vector = std::vector<double>;
using VectorForClasses = std::vector<int>;
using MTLVector = mtl::dense_vector<double>;
using MTLMatrix = std::vector<MTLVector>;
using NativeMatrixForDistance = std::vector<Vector>;
using NativeMatrixForIndex = std::vector<std::vector<int>>;
using MatriciesPair = std::pair<NativeMatrixForDistance, NativeMatrixForIndex>;


enum class MetricType
{
  kOverlap,
  kFlattenedOverlap,
  kLogOverlap,
};

class Overlap{};
class FlattenedOverlap{};
class kLogOverlap{};

enum class OperandType
{
  kX,
  kZ
};

struct NearestNeighbours
{
  static constexpr size_t kPoolSize = 15;

private:
  using Metric = double(NearestNeighbours *, const MTLVector, const MTLVector);

public:
  NearestNeighbours(size_t n_neighbours,
                    MetricType metric_type);

  void fit(const Vector &X, const VectorForClasses &Y, const size_t amount_of_features);

  MatriciesPair kneighbors(const Vector &v_X, const ProbabilitiesMapTensor &p_2_matrix);
  MatriciesPair kneighbors(const Vector &v_X);

private:
  MTLVector get_frequency_vector(const MTLVector &x);
  MTLVector get_parameter_vectors(const MTLVector &x);
  void set_parameter_tensor(const MTLMatrix &X, const ProbabilitiesMapTensor *tensor);

  void set_params_for_flattened_overlap(const ProbabilitiesMapTensor &matrix);
  void set_params_for_log_overlap(const ProbabilitiesMapTensor &matrix);

  template <typename T, typename IndexType>
  void sort_both(std::vector<T> &v, std::vector<IndexType> &indexies);

  template <typename T, typename IndexType>
  void shrink_both(std::vector<T> &v, std::vector<IndexType> &indexies);

  double overlap(const MTLVector &x, const MTLVector &z);
  double flattened_overlap(const MTLVector &x, const MTLVector &z);
  double log_overlap(const MTLVector &x, const MTLVector &z);

private:
  MetricType metric_type;
  MTLMatrix X_train;
  MTLVector Y_train;
  size_t amount_of_features = 0;
  size_t n_neighbours;


  const ProbabilitiesMapTensor *X_parameter_tensor;
  const ProbabilitiesMapTensor *Z_parameter_tensor;




  ThreadPoolPtr pool;

private:
  std::map<MetricType, std::function<Metric>> metrics = {
      {MetricType::kOverlap, &NearestNeighbours::overlap},
      {MetricType::kLogOverlap, &NearestNeighbours::log_overlap},
      {MetricType::kFlattenedOverlap, &NearestNeighbours::flattened_overlap},
  };
};

template <typename ResultType = MTLVector, typename V1, typename V2>
ResultType operator*(const V1 &lhs, const V2 &rhs);
template <typename ResultType = MTLVector, typename V1, typename V2>
ResultType operator+(const V1 &lhs, const V2 &rhs);

template <typename ValueType = bool>
mtl::dense_vector<ValueType> equal(const MTLVector &lhs, const MTLVector &rhs);

template <typename ValueType = bool>
mtl::dense_vector<ValueType> not_equal(const MTLVector &lhs, const MTLVector &rhs);

template <typename ValueType, typename ReturnType = ValueType>
mtl::dense_vector<ReturnType> get_mtl_vector(const std::vector<ValueType> &stl_vector_const);

MTLMatrix get_mtl_matrix(const Vector &v_matrix_const, size_t m, size_t n);

// class
#ifndef __INCLUDE_LIBS
#include "regress.tpp"
#endif
