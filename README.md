
This is a common project between the department of Systems Immunology (SIMM) of the he Helmholtz Center for Infection Research (HZI) and the Institute for Software Technology of the German Aerospace Center (DLR). This project will bring cutting edge and compute intensive epidemiological models to a large scale, which enables a precise and high-resolution spatiotemporal pandemic simulation for entire countries.


**Getting started**

This project is divided into multiple building blocks. The implementation of the epidemiological models is to be found in cpp. Data acquisition tools and data is to be found in data. The interactive frontend is to be found under frontend. It is regularly deployed to http://hpcagainstcorona.sc.bs.dlr.de/index.html. In pycode you find python bindings to call the C++ code available in cpp. At the moment, some data tools are still under pycode, too.


**Requirements**

…


**Installation** 

*Making and executing C++ code*

* (Create a build folder and) do cmake .. in epidemiology-cpp/cpp/build
* Do cmake --build .
* Run 
  * an example via ./examples/secir_ageres
  * all unit tests via ./tests/runUnitTests

*Steps to execute C++ code via python bindings*

*  Create a python virtual environment via python3 -m venv virtualenv/
*  Activate the environment via source virtualenv/bin/activate
*  Do pip3 install scikit-build
*  In epidemiology-cpp/pycode do
   *  python3 setup.py build
   *  python3 setup.py install
   *  execute some example


**Development**
* [Git workflow and change process](https://gitlab.dlr.de/hpc-against-corona/epidemiology/-/wikis/Git-workflow-and-change-process)
* [C++ Coding Guidelines](https://gitlab.dlr.de/hpc-against-corona/epidemiology/-/wikis/Cpp-Coding-Guidlines)
* [Python Coding Guidelines](https://gitlab.dlr.de/hpc-against-corona/epidemiology/-/wikis/Python%20Coding%20Guidelines)