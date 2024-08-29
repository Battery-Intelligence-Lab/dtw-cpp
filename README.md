DTW-C++
===========================
[![Ubuntu unit](https://github.com/Battery-Intelligence-Lab/dtw-cpp/workflows/Ubuntu%20unit/badge.svg)](https://github.com/Battery-Intelligence-Lab/dtw-cpp/actions)
[![macOS unit](https://github.com/Battery-Intelligence-Lab/dtw-cpp/workflows/macOS%20unit/badge.svg)](https://github.com/Battery-Intelligence-Lab/dtw-cpp/actions)
[![Windows unit](https://github.com/Battery-Intelligence-Lab/dtw-cpp/workflows/Windows%20unit/badge.svg)](https://github.com/Battery-Intelligence-Lab/dtw-cpp/actions)
![Website](https://img.shields.io/website?url=https%3A%2F%2FBattery-Intelligence-Lab.github.io%2Fdtw-cpp%2F)
[![codecov](https://codecov.io/gh/Battery-Intelligence-Lab/dtw-cpp/branch/main/graph/badge.svg?token=K739SRV4QG)](https://codecov.io/gh/Battery-Intelligence-Lab/dtw-cpp)

![Contributors](https://img.shields.io/github/contributors/Battery-Intelligence-Lab/dtw-cpp)
![Last update](https://img.shields.io/github/last-commit/Battery-Intelligence-Lab/dtw-cpp/develop)
![Issues](https://img.shields.io/github/issues/Battery-Intelligence-Lab/dtw-cpp)
![Forks](https://img.shields.io/github/forks/Battery-Intelligence-Lab/dtw-cpp)
![Stars](https://img.shields.io/github/stars/Battery-Intelligence-Lab/dtw-cpp)

![GitHub all releases](https://img.shields.io/github/downloads/Battery-Intelligence-Lab/dtw-cpp/total) 
[![](https://img.shields.io/badge/license-BSD--3--like-5AC451.svg)](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/LICENSE)

There is separate [detailed documentation](https://Battery-Intelligence-Lab.github.io/dtw-cpp/) available for this project; this `readme.md` file only gives a short summary. 

Introduction
===========================
DTW-C++ is a dynamic time warping (DTW) and clustering library for time series data, written in C++. Users can input multiple time series and find clusters of similar time series. Time series may be the same lengths, or different lengths. The number of clusters to find may be fixed, or a range of numbers to try may be specified. DTW-C++ finds clusters in time series data using k-medoids or mixed integer programming (MIP); k-medoids is generally faster, but may get stuck in local optima; MIP can find globally optimal clusters.
<p align="center"><img src="./media/Merged_document.png" alt="DTW" width="60%"/></center></p>

Contributors
===========================
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section --><!-- prettier-ignore-start --><!-- markdownlint-disable -->
<table>
	<tbody>
		<tr>
			<td style="text-align:center; vertical-align:top"><a href="https://github.com/beckyperriment"><img alt="Becky Perriment" src="https://avatars.githubusercontent.com/u/93582518?v=4?s=100" style="width:100px" /><br />
			<sub><strong>Becky Perriment</strong></sub></a><br />
			<a href="https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/develop/contributors.md#core-contributors">💡💻👀⚠️</a></td>
			<td style="text-align:center; vertical-align:top"><a href="https://github.com/ElektrikAkar"><img alt="Volkan Kumtepeli" src="https://avatars.githubusercontent.com/u/8674942?v=4?s=100" style="width:100px" /><br />
			<sub><strong>Volkan Kumtepeli</strong></sub></a><br />
			<a href="https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/develop/contributors.md#core-contributors">💡💻👀⚠️🚇🐢</a></td>
			<td style="text-align:center; vertical-align:top"><a href="http://howey.eng.ox.ac.uk"><img alt="David Howey" src="https://avatars.githubusercontent.com/u/2247552?v=4?s=100" style="width:100px" /><br />
			<sub><strong>David Howey</strong></sub></a><br />
			<a href="https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/develop/contributors.md#core-contributors">💡👀</a></td>
		</tr>
	</tbody>
</table>
<!-- markdownlint-restore --><!-- prettier-ignore-end --><!-- ALL-CONTRIBUTORS-LIST:END -->
