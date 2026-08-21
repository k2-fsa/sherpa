.. _onnx-download-qnn:

Download QNN SDK
================

This section describes how to download the QNN SDK.

Please go to

  `<https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community>`_

and select a version to download.

.. image:: ./pic/select-qnn.jpg
   :align: center
   :alt: screenshot of selecting a version of QNN to download
   :width: 600

We’re using version 2.40.0.251030, but you can choose whichever version you prefer.

After selecting the version to download, just click the button ``Download``, or use
the following URL to download it::

  https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.40.0.251030/v2.40.0.251030.zip

Assume we want to download it to the directory ``$HOME/qnn``, just do::

  mkdir $HOME/qnn
  cd $HOME/qnn
  wget https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.40.0.251030/v2.40.0.251030.zip

  unzip v2.40.0.251030.zip

After unzipping, you will get the following directory::

  $HOME/qnn/qairt/2.40.0.251030

.. code-block::

  ls -lh qnn/qairt/2.40.0.251030/

  total 752K
  -rwxrwxrwx  1 kuangfangjun root  788 Oct 30 21:57 GENIE_README.txt
  -rwxrwxrwx  1 kuangfangjun root 145K Oct 30 21:57 LICENSE.pdf
  -rwxrwxrwx  1 kuangfangjun root 127K Oct 30 21:57 NOTICE.txt
  -rwxrwxrwx  1 kuangfangjun root  75K Oct 30 21:58 NOTICE_WINDOWS.txt
  -rwxrwxrwx  1 kuangfangjun root 223K Oct 30 21:57 QAIRT_ReleaseNotes.txt
  -rwxrwxrwx  1 kuangfangjun root 127K Oct 30 21:57 QNN_NOTICE.txt
  -rwxrwxrwx  1 kuangfangjun root 1.4K Oct 30 21:57 QNN_README.txt
  -rwxrwxrwx  1 kuangfangjun root  45K Oct 30 21:58 QNN_TFLITE_DELEGATE_NOTICE.txt
  -rwxrwxrwx  1 kuangfangjun root 1.1K Oct 30 21:58 QNN_TFLITE_DELEGATE_README.txt
  -rwxrwxrwx  1 kuangfangjun root 6.8K Oct 30 21:58 QNN_TFLITE_DELEGATE_ReleaseNotes.txt
  drwxrwxrwx  4 kuangfangjun root    0 Oct 30 21:57 benchmarks
  drwxrwxrwx 19 kuangfangjun root    0 Oct 30 21:58 bin
  drwxrwxrwx  5 kuangfangjun root    0 Oct 30 21:57 docs
  drwxrwxrwx  7 kuangfangjun root    0 Oct 30 21:57 examples
  drwxrwxrwx  5 kuangfangjun root    0 Oct 30 21:57 include
  drwxrwxrwx 22 kuangfangjun root    0 Oct 30 21:58 lib
  -rwxrwxrwx  1 kuangfangjun root  982 Oct 30 21:57 sdk.yaml
  drwxrwxrwx  4 kuangfangjun root    0 Oct 30 21:58 share

QNN_SDK_ROOT
------------

Now you can run::

  export QNN_SDK_ROOT=$HOME/qnn/qairt/2.40.0.251030
