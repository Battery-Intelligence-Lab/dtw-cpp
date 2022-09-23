/*
 * Range.hpp
 *
 *  A basic Range class not to create vector for Iota
 *
 *  Created on: 18 Aug 2022
 *      Author: Volkan Kumtepeli
 */

#pragma once

#include <iterator>

namespace dtwc {

struct RangeIterator
{
  /*
    Adapted from: https://www.internalpointers.com/post/writing-custom-iterators-modern-cpp
    https://stackoverflow.com/questions/12092448/code-for-a-basic-random-access-iterator-based-on-pointers
  */
  using value_type = int;
  using iterator_category = std::random_access_iterator_tag; // Something is missing.
  using difference_type = value_type;

  using pointer = value_type *;
  using reference = value_type &;

  value_type _n{ 0 };

  explicit RangeIterator(value_type i) : _n(i) {}
  RangeIterator(const RangeIterator &rhs) : _n(rhs._n) {}

  reference operator*() { return _n; }
  pointer operator->() { return &_n; }

  // Prefix increment
  inline RangeIterator &operator++()
  {
    _n++;
    return *this;
  }

  inline RangeIterator &operator--()
  {
    --_n;
    return *this;
  }

  // Postfix increment
  inline RangeIterator operator++(int)
  {
    RangeIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  inline RangeIterator operator--(int)
  {
    RangeIterator tmp(*this);
    --_n;
    return tmp;
  }

  friend inline bool operator==(const RangeIterator &a, const RangeIterator &b) { return a._n == b._n; };
  friend inline bool operator!=(const RangeIterator &a, const RangeIterator &b) { return a._n != b._n; };

  inline RangeIterator &operator+=(difference_type rhs)
  {
    _n += rhs;
    return *this;
  }
  inline RangeIterator &operator-=(difference_type rhs)
  {
    _n -= rhs;
    return *this;
  }

  inline value_type operator[](difference_type rhs) { return rhs; }


  inline difference_type operator-(const RangeIterator &rhs) const { return _n - rhs._n; }
  inline RangeIterator operator+(difference_type rhs) const { return RangeIterator(_n + rhs); }
  inline RangeIterator operator-(difference_type rhs) const { return RangeIterator(_n - rhs); }
  friend inline RangeIterator operator+(difference_type lhs, const RangeIterator &rhs) { return RangeIterator(lhs + rhs._n); }
  friend inline RangeIterator operator-(difference_type lhs, const RangeIterator &rhs) { return RangeIterator(lhs - rhs._n); }

  // inline bool operator==(const RangeIterator &rhs) const { return _n == rhs._n; }
  // inline bool operator!=(const RangeIterator &rhs) const { return _n != rhs._n; }
  // inline bool operator>(const RangeIterator &rhs) const { return _n > rhs._n; }
  // inline bool operator<(const RangeIterator &rhs) const { return _n < rhs._n; }
  // inline bool operator>=(const RangeIterator &rhs) const { return _n >= rhs._n; }
  // inline bool operator<=(const RangeIterator &rhs) const { return _n <= rhs._n; }
};

class Range
{
  int n_begin{ 0 }, n_end{ 0 };

public:
  explicit Range(int i) : n_end(i) {}
  Range(int i_begin, int i_end) : n_begin(i_begin), n_end(i_end) {}

  RangeIterator begin() { return RangeIterator(n_begin); }
  RangeIterator end() { return RangeIterator(n_end); }
};

} // namespace dtwc