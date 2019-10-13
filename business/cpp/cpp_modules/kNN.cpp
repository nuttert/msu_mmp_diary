#include <iostream>
#include <math.h>
#include <algorithm>
#include "kNN.hpp"



Element::Element(double x,double y):x(x),y(y){};

Element::Element(const std::vector<double>&list){
  x = list[0];
  y = list[1];
}
TrainElement::TrainElement(double x,double y, size_t class_number):Element(x,y),class_number(class_number){};


ClassDistance TestElement::EuclideDistance(const TrainElement& rhs) const{
  return {sqrt(pow((x-rhs.x),2) + pow((y-rhs.y),2)), rhs.class_number};
}

std::vector<ClassDistance> TestElement::EuclideDistance(const TrainElements& rhs) const{
  std::vector<ClassDistance> distancies{};
  for(const auto& element:rhs){
      const auto distance = EuclideDistance(element);
      distancies.push_back(distance);
  }
  return distancies;
}



std::vector<size_t> classifyKnn(const TrainElements& train_data, const TestElements& test_data,size_t k,size_t amount_of_classes){
    std::vector<size_t> Y{};

    for(const auto& test_element:test_data){
        auto distancies = test_element.EuclideDistance(train_data);
        std::sort(distancies.begin(), distancies.end());
        std::vector<size_t> statistics(amount_of_classes,0);
        for(size_t index = 0;index < std::min(k, distancies.size());++index){
          const auto[distance, class_number] = distancies[index];
          ++statistics[class_number];
        }
        const auto max_element_ = std::max_element(statistics.begin(), statistics.end());
        const auto class_number =  max_element_ - statistics.begin();
        Y.push_back(class_number);
    }
    return Y;
}
