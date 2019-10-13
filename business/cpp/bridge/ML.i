%module ML

%include "std_multiset.i"

%{  
    #include "cpp_modules/kNN.hpp"
    #include "cpp_modules/matrix.hpp"
%}


%include "std_vector.i"
%include <std_shared_ptr.i>

namespace std {
    %template(IntVector) vector<size_t>;
    %template(vectord) vector<double>;
}

namespace std {
    %template(ElementVector) vector<Element>;
    %template(TrainElementVector) vector<TrainElement>;
    %template(TestElementVector) vector<TestElement>;
}

#define __INCLUDE_LIBS
%include "cpp_modules/kNN.hpp"
%include "cpp_modules/matrix.hpp"

%template(DoubleMatrix) Matrix<double>;
%template(BoolMatrix) Matrix<bool>;
