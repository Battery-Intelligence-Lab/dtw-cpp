/*
 * unit_test_file_io.cpp
 *
 *  Created on: 04 Dec 2022
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#include "../../src/dtwc.hpp"


namespace dtwc::tests::unit {

bool io_tsv_read()
{

  dtwc::DataLoader dl(settings::root_folder / "data/benchmark/UCRArchive_2018/UMD/UMD_TEST.tsv");

  dl.startColumn(1);
  auto data = dl.load();

  if (data.p_names[0] != "0")
    return false;

  return true;
}

bool io_csv_read()
{


  return true;
}

bool io_folder_read()
{
  // From Pandas:
  dtwc::DataLoader dl;

  dl.path(settings::root_folder / "data/dummy").startRow(1).startColumn(1);
  auto data = dl.load();

  return true;
}


int test_all_file_io()
{
  if (!io_tsv_read()) return 1;
  if (!io_csv_read()) return 2;
  if (!io_folder_read()) return 3;

  return 0;
}


} // namespace dtwc::tests::unit


int main() { return dtwc::tests::unit::test_all_file_io(); }