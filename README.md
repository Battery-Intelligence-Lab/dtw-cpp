DTW-C++
===========================
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06881/status.svg)](https://doi.org/10.21105/joss.06881)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13551469.svg)](https://doi.org/10.5281/zenodo.13551469)
[![Website](https://img.shields.io/website?url=https%3A%2F%2FBattery-Intelligence-Lab.github.io%2Fdtw-cpp%2F)](https://Battery-Intelligence-Lab.github.io/dtw-cpp/)



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
DTW-C++ is a C++ library for dynamic time warping (DTW) and clustering of time series data. Users can input multiple time series and find clusters of similar time series. The time series can have the same or different lengths. The number of clusters to find can be fixed or specified as a range to try. DTW-C++ finds clusters in time series data using k-medoids or mixed integer programming (MIP). K-medoids is generally faster but may get stuck in local optima, while MIP can give guarantees about globally optimal clusters.
<p align="center"><img src="./media/Merged_document.png" alt="DTW" width="60%"/></center></p>

Citation
===========================

APA style: 
```
Kumtepeli, V., Perriment, R., & Howey, D. A. (2024). DTW-C++: Fast dynamic time warping and clustering of time series data. Journal of Open Source Software, 9(101), 6881. https://doi.org/10.21105/joss.06881
```

BibTeX: 
```
@article{Kumtepeli2024,
author = {Kumtepeli, Volkan and Perriment, Rebecca and Howey, David A.},
doi = {10.21105/joss.06881},
journal = {Journal of Open Source Software},
month = sep,
number = {101},
pages = {6881},
title = {{DTW-C++: Fast dynamic time warping and clustering of time series data}},
url = {https://joss.theoj.org/papers/10.21105/joss.06881},
volume = {9},
year = {2024}
}
```

<div class="tabs">
    <input type="radio" id="tab1" name="tabs" checked>
    <label for="tab1">MLA</label>
    <div class="tab">
        <pre>
Author's Last Name, First Name. "Title of Article." Title of Journal, vol. #, no. #, Publication Date, pp. Page Range. Database Name, DOI or URL.
        </pre>
    </div>

    <input type="radio" id="tab2" name="tabs">
    <label for="tab2">APA</label>
    <div class="tab">
        <pre>
Author's Last Name, F. M. (Year). Article title. Journal Title, Volume(Issue), Page Range. https://doi.org/DOI
        </pre>
    </div>

    <input type="radio" id="tab3" name="tabs">
    <label for="tab3">Chicago</label>
    <div class="tab">
        <pre>
Author's Last Name, First Name. "Article Title." Journal Title Volume, no. Issue (Year): Page Range. https://doi.org/DOI.
        </pre>
    </div>

    <input type="radio" id="tab4" name="tabs">
    <label for="tab4">Harvard</label>
    <div class="tab">
        <pre>
Author's Last Name, First Initial(s). Year. Title of Paper. Title of Journal, Volume Number(Issue Number), Page Numbers.
        </pre>
    </div>

    <input type="radio" id="tab5" name="tabs">
    <label for="tab5">Vancouver</label>
    <div class="tab">
        <pre>
Author(s). Title of article. Abbreviated Journal Title. Year;Volume(Issue):Page numbers.
        </pre>
    </div>

    <input type="radio" id="tab6" name="tabs">
    <label for="tab6">IEEE</label>
    <div class="tab">
        <pre>
Author's Last Name, First Initial(s), "Title of Article," Journal Title, vol. Volume Number, no. Issue Number, pp. Page Numbers, Year.
        </pre>
    </div>

    <input type="radio" id="tab7" name="tabs">
    <label for="tab7">BibTeX</label>
    <div class="tab">
        <pre>
@article{citation_key,
  author = {Author's Last Name, First Name},
  title = {Title of Article},
  journal = {Journal Title},
  year = {Year},
  volume = {Volume},
  number = {Issue},
  pages = {Page Range},
  url = {DOI or URL}
}
        </pre>
    </div>
</div>


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
