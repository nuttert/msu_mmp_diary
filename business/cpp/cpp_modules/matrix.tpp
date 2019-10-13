#include "matrix.hpp"


template<typename ElementType>
struct Matrix<ElementType>::BaseMatrix: public mtl::dense2D<ElementType>{
  using Base = mtl::dense2D<ElementType>;
  using mtl::dense2D<ElementType>::dense2D;
  using mtl::dense2D<ElementType>::operator=;
};

template<typename ElementType>
Matrix<ElementType>::Matrix(size_t n, size_t m){
  matrix_impl_ = std::make_shared<BaseMatrix>(n,m);
}


template<typename ElementType>
Matrix<ElementType> Matrix<ElementType>::operator+(Matrix& matrix)const{
  using namespace mtl;
   Matrix result{matrix.matrix_impl_->dim1(), matrix.matrix_impl_->dim2()};
  *result.matrix_impl_ = *matrix.matrix_impl_ + *matrix_impl_;
  return result;
}

template<typename ElementType>
Matrix<ElementType> Matrix<ElementType>::operator-(const Matrix& matrix)const{
   Matrix result{matrix.matrix_impl_->dim1(), matrix.matrix_impl_->dim2()};
  *result.matrix_impl_ = *matrix.matrix_impl_ - *matrix_impl_;
  return result;
}

template<typename ElementType>
Matrix<ElementType> Matrix<ElementType>::operator*(const Matrix& matrix)const{
  using BaseMatrix = Matrix::BaseMatrix;
  using Base = typename BaseMatrix::Base;

  Matrix result{matrix.matrix_impl_->dim1(), matrix.matrix_impl_->dim2()};
  
  const Base& first_matrix = *matrix.matrix_impl_;
  const Base& second_matrix = *matrix_impl_;

  *result.matrix_impl_ = first_matrix * second_matrix;
  return result;
}

template<typename ElementType>
Matrix<ElementType>& Matrix<ElementType>::operator=(const Matrix& matrix){
  *matrix_impl_ = *matrix.matrix_impl_;
  return *this;
}

template<typename ElementType>
Matrix<ElementType>& Matrix<ElementType>::clear(){
  *matrix_impl_ = 0;
  return *this;
}

template<typename ElementType>
void Matrix<ElementType>::print(){
  std::cout << *matrix_impl_;
}

template<typename ElementType>
ElementType& Matrix<ElementType>::operator()(size_t i,size_t j){
  return (*matrix_impl_)(i,j);
}
