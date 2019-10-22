%module ML

%include "std_multiset.i"
%include "std_multimap.i"

%{  
    #include "cpp_modules/regress.hpp"
%}


%include "std_vector.i"
%include <std_shared_ptr.i>
%include <std_pair.i>

namespace std {


    %template(IntDoubleMap) map<int,double>;
    %template(DoubleDoubleMap) map<double,double>;

    %template(MapVectorDoubleDouble) vector<map<double,double>>;
    %template(DoubleVector) vector<double>;
    %template(IntVector) vector<int>;
    %template(ShrdPtrToDoubleVector) shared_ptr<vector<double>>;
    %template(NativeMatrixPy) vector<vector<double>>;
    %template(NativeMatrixForIndexPy) vector<vector<int>>;
    %shared_ptr(std::vector<std::map<int, double>>);
    %template(TensorProb) std::shared_ptr<std::vector<std::map<int, double>>>;
    %template(MapVectorIntDouble) vector<map<int,double>>;
    %template(ReturnedMatrix) pair<vector<vector<double>>, vector<vector<int>>>;
}


#define __INCLUDE_LIBS
%include "cpp_modules/regress.hpp"

