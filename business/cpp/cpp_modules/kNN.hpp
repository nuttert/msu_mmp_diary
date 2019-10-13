#pragma once
#ifndef __INCLUDE_LIBS
#include <vector>
#include <memory>
#endif

struct ClassDistance{
    double distance;
    size_t class_number;

    bool operator <(const ClassDistance& rhs)const{ return distance < rhs.distance;}
    bool operator <=(const ClassDistance& rhs)const{ return distance <= rhs.distance;}
    bool operator ==(const ClassDistance& rhs)const{ return distance == rhs.distance;}
    bool operator !=(const ClassDistance& rhs)const{ return distance != rhs.distance;}
    bool operator >(const ClassDistance& rhs)const{ return distance > rhs.distance;}
};



struct Element{
    double x,y;
    Element() = default;
    Element(double x,double y);
    Element(const std::vector<double>& list);
};

struct TrainElement:Element{
    size_t class_number;
    TrainElement() = default;
    TrainElement(double x,double y, size_t class_number);
    TrainElement(const std::vector<double>& list):Element(list){};
};

struct TestElement: public Element{
    using TrainElements = std::vector<TrainElement>;
    TestElement() = default;
    TestElement(double x,double y):Element(x,y){}
    TestElement(const std::vector<double>& list):Element(list){};
    TestElement(const TrainElement& train_el):Element(train_el){};

    ClassDistance EuclideDistance(const TrainElement& rhs) const;
    std::vector<ClassDistance> EuclideDistance(const TrainElements& rhs) const;
};

using TrainElements = std::vector<TrainElement>;
using TestElements = std::vector<TestElement>;


std::vector<size_t> classifyKnn(const std::vector<TrainElement>& train_data, const std::vector<TestElement>& test_data,size_t k,size_t amount_of_classes);
