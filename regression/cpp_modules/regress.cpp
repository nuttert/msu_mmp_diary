#include "regress.hpp"
#include "boost/thread/thread_pool.hpp"
#include <mutex>


// void async_operation_handler(){

// }


MTLMatrix get_mtl_matrix(const Vector &v_matrix_const, size_t m, size_t n)
{
  auto &v_matrix = const_cast<Vector &>(v_matrix_const);
  MTLMatrix matrix(m);
  size_t start_index = 0;

  for (size_t i = 0; i < m; ++i)
  {
    auto pointer = &v_matrix[start_index];
    matrix[i] = MTLVector(n, pointer);
    start_index += n;
  }
  return matrix;
}

NearestNeighbours::NearestNeighbours(size_t n_neighbours,
                                     MetricType metric_type) : n_neighbours(n_neighbours),
                                                               metric_type(metric_type)
{
  pool = std::make_shared<ThreadPool>(kPoolSize);
}

void NearestNeighbours::fit(const Vector &X, const VectorForClasses &Y, const size_t amount_of_features)
{
  this->amount_of_features = amount_of_features;

  const size_t m = X.size() / amount_of_features;
  const size_t n = amount_of_features;

  X_train = get_mtl_matrix(X, m, amount_of_features);
  Y_train = get_mtl_vector(Y);
}

double NearestNeighbours::flattened_overlap(
    const MTLVector &x,
    const MTLVector &z)
{
  const MTLVector x_eq_z = equal<double>(x, z);
  const MTLVector x_not_eq_z = not_equal<double>(x, z);
  auto weights_v = Vector(size(x));

  for (size_t index = 0; index < size(x); ++index)
  {
    const auto &x_i = x[index];
    const auto &p_2_column = (*X_parameter_tensor)[index];
    auto weight_iterator = p_2_column.find(x_i);
    if (weight_iterator != p_2_column.end())
      weights_v[index] = weight_iterator->second;
    else
      weights_v[index] = 0;
  }
  auto weights = get_mtl_vector(weights_v);
  return mtl::sum(weights * x_not_eq_z + x_not_eq_z);
}

double NearestNeighbours::overlap(const MTLVector &x, const MTLVector &z)
{
  return mtl::sum(not_equal<int>(x, z));
}

double NearestNeighbours::log_overlap(const MTLVector &x, const MTLVector &z)
{
  MTLVector FZ = get_parameter_vectors(z);
  MTLVector FX = get_parameter_vectors(x);


  const auto x_not_eq_z = not_equal<double>(x, z);
  auto weights = log(add(FX, 1.)) * log(add(FZ, 1.));

  return mtl::sum(weights * x_not_eq_z);
}

MatriciesPair NearestNeighbours::kneighbors(const Vector &v_X, const ProbabilitiesMapTensor &matrix)
{
  switch (metric_type)
  {
  case MetricType::kFlattenedOverlap:
    set_params_for_flattened_overlap(matrix);
    break;
  case MetricType::kLogOverlap:
    set_params_for_log_overlap(matrix);
    break;
  default:
    break;
  }
  return kneighbors(v_X);
}

MatriciesPair NearestNeighbours::kneighbors(const Vector &v_X)
{
  const size_t m = v_X.size() / amount_of_features;
  const size_t n = amount_of_features;
  const size_t m_train = size(X_train);

  auto &metric = metrics[metric_type];

  auto result_distance = NativeMatrixForDistance(m, Vector(m_train));
  auto result_indexies = NativeMatrixForIndex(m, std::vector<int>(m_train, 0));
  const auto X = get_mtl_matrix(v_X, m, n);
  for (size_t i = 0; i < m; ++i)
  {

    boost::asio::post(*pool,
                      [i,
                       &X,
                       &result_distance,
                       &result_indexies,
                       &metric,
                       &m_train,
                       this] {
                        const auto &x = X[i];
                        for (size_t j = 0; j < m_train; ++j)
                        {
                          const auto &z = X_train[j];
                          result_distance[i][j] = metric(this, x, z);
                        }
                        std::cout << "i:" << i << std::endl;
                        sort_both(result_distance[i], result_indexies[i]);
                        shrink_both(result_distance[i], result_indexies[i]);
                      });
  }
  pool->join();
  return {result_distance, result_indexies};
}

MTLVector NearestNeighbours::get_parameter_vectors(const MTLVector &x)
{
  switch (metric_type)
  {
  case MetricType::kLogOverlap:
    return std::move(get_frequency_vector(x));
    break;
  default:
    break;
  }
  return {};
}



MTLVector NearestNeighbours::get_frequency_vector(const MTLVector &x)
{

  if (size(x) != Z_parameter_tensor->size())
    throw std::runtime_error("Size error");

  auto Frequency = MTLVector(size(x));
  for (size_t index = 0; index < mtl::size(x); ++index)
  {
    const auto &current_frequency_column = (*Z_parameter_tensor)[index];
    const auto &x_i = x[index];

    const auto result_i_iterator = current_frequency_column.find(x_i);
    if (result_i_iterator != current_frequency_column.end())
      Frequency[index] = result_i_iterator->second;
    else
      Frequency[index] = 0;
  }
  return Frequency;
}

void NearestNeighbours::set_params_for_flattened_overlap(const ProbabilitiesMapTensor &matrix)
{
  const auto size = matrix.size();
  const auto p_2_map_sum_matrix = new ProbabilitiesMapTensor(size);

  for (size_t column_number = 0; column_number < matrix.size(); ++column_number)
  {
    auto &p_2_column = matrix[column_number];
    auto column_for_sort = ProbabilityVectorMatrix(p_2_column.begin(), p_2_column.end());

    sort(column_for_sort.begin(), column_for_sort.end(),
         [](auto p1, auto p2) { return p1.second < p2.second; });

    double sum = 0;
    for (auto begin = column_for_sort.begin(),
              end = column_for_sort.end();
         begin != end; ++begin)
    {
      auto next_iterator = begin.operator+(1);
      begin->second = sum += begin->second;
      while (next_iterator != end &&
             begin->second == next_iterator->second)
      {
        ++next_iterator;
        begin->second += next_iterator->second;
      }
    }
    auto &p_2_column_self = (*p_2_map_sum_matrix)[column_number];
    p_2_column_self = ProbabilityMapMatrix(column_for_sort.begin(), column_for_sort.end());
  }
  this->X_parameter_tensor = p_2_map_sum_matrix;
}

void NearestNeighbours::set_params_for_log_overlap(const ProbabilitiesMapTensor &Z_parameter_tensor)
{
  this->Z_parameter_tensor = &Z_parameter_tensor;
}
