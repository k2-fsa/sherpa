.. list-table:: Comparison of ``libmodel.so`` and ``model.bin``
   :header-rows: 1

   * - Feature
     - libmodel.so
     - model.bin
   * - OS Dependency
     - | **OS-dependent**: cannot run across
       | different OS/arch
       | (e.g., Android/arm64
       | vs Linux/arm64)
     - | **OS-independent**: can run on
       | multiple OS/arch
       | (e.g., Android/arm64
       | and Linux/arm64)
   * - SoC Dependency
     - | **SoC-independent**: can run
       | on multiple Qualcomm chips
       | (e.g., SM8850, SA8259, QCS9100)
     - | **SoC-dependent**: built for
       | a specific chip;
       | cannot run on a different SoC
   * - QNN-SDK Dependency
     - | **QNN-SDK-independent**: works
       | with any QNN SDK version
     - | **QNN-SDK-dependent**: depends
       | on the QNN SDK version
       | used to build it
   * - First-Run Initialization
     - | **Slow**: context must be
       | generated at runtime
     - | **Fast**: context is
       | pre-generated
   * - Recommended Use
     - | When SoC-independence or
       | SDK-independence is needed
     - When fastest startup is required

**Note:** Choose ``libmodel.so`` if you need flexibility across SoCs or
QNN SDK versions. Use ``model.bin`` if you want the fastest possible
first-run initialization on a specific SoC.

