#pragma once
#ifndef __INCLUDE_LIBS
#include <memory>
#include <boost/numeric/mtl/mtl.hpp>
#endif



template<typename ElementType>
struct Matrix{

struct BaseMatrix;
Matrix(size_t n, size_t m);
Matrix operator+(Matrix& matrix)const;
Matrix operator-(const Matrix& matrix)const;
Matrix operator*(const Matrix& matrix)const;
Matrix& operator=(const Matrix& matrix);
ElementType& operator()(size_t i,size_t j);
Matrix& clear();
void print();
private:
std::shared_ptr<BaseMatrix> matrix_impl_;
};



#include"matrix.tpp"
