# Third-party notices

The project's original code and associated documentation are distributed under
the [MIT License](LICENSE). Third-party components retain their own licenses.

* **Plotly.js 2.35.0** is bundled in `workbench/vendor/plotly.min.js` for offline
  plotting. Its MIT notice is in `workbench/vendor/plotly-LICENSE.txt`, copied
  from the [upstream 2.35.0 license](https://github.com/plotly/plotly.js/blob/v2.35.0/LICENSE).
* **FMMAX** is acknowledged in the Fourier-factorization and eigendecomposition
  implementations for the constructions they follow. Its MIT notice is retained
  in `licenses/fmmax-LICENSE.txt`; see the
  [upstream project](https://github.com/facebookresearch/fmmax).
* **TORCWA** is an optional, separately installed comparison engine. It is not
  vendored here. References to its conventions and Lorentzian gradient method
  are methodological acknowledgments. TORCWA's own code is licensed under
  [LGPL-3.0-or-later](https://github.com/kch3782/torcwa/blob/main/LICENSE);
  this repository's MIT license does not replace that license.
* **Ansys Lumerical** is an optional external proprietary application. Its
  executable, Python API and license are not included.
* PyTorch, NumPy, SciPy, Matplotlib and the development tools are installed
  separately and retain their respective licenses.

The historical research sources under `paper/archive` are snapshots from this
project, including comparison adapters that import external packages. They do
not bundle those external solvers. The archive inventory records their origin
and content hashes.
