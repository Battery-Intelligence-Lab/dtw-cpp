/**
 * @file Index.hpp
 * @brief A basic Index class for Range class
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 18 Aug 2022
 */

#pragma once

#include <iterator>

namespace dtwc {

class Index
{
  /*
  Adapted from: https://stackoverflow.com/questions/61208870/how-to-write-a-random-access-custom-iterator-that-can-be-used-with-stl-algorithm
*/
  size_t ptr{};

public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = size_t;
  using reference = size_t;
  using pointer = size_t;
  using difference_type = size_t;

  Index() = default;
  Index(size_t ptr) : ptr(ptr) {}

  reference operator*() const { return ptr; }
  pointer operator->() const { return ptr; }

  Index &operator++()
  {
    ptr++;
    return *this;
  }
  Index &operator--()
  {
    ptr--;
    return *this;
  }

  difference_type operator-(const Index &it) const { return this->ptr - it.ptr; }

  Index operator+(const difference_type &diff) const { return Index(ptr + diff); }
  Index operator-(const difference_type &diff) const { return Index(ptr - diff); }

  reference operator[](const difference_type &offset) const { return *(*this + offset); }

  bool operator==(const Index &it) const { return this->ptr == it.ptr; }
  bool operator!=(const Index &it) const { return this->ptr != it.ptr; }
  bool operator<(const Index &it) const { return this->ptr < it.ptr; }
  bool operator>(const Index &it) const { return this->ptr > it.ptr; }
  bool operator>=(const Index &it) const { return !(this->ptr < it.ptr); }
  bool operator<=(const Index &it) const { return !(this->ptr > it.ptr); }
};

} // namespace dtwc