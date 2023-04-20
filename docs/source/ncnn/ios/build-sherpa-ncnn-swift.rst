Build sherpa-ncnn for iOS
=========================

This section describes how to build `sherpa-ncnn`_ for ``iPhone`` and ``iPad``.

Requirement
-----------

.. warning::

  The minimum deployment requires the iOS version ``>= 13.0``.


Before we continue, please make sure the following requirements are satisfied:

- macOS. It won't work on Windows or Linux.
- Xcode. The version ``14.2 (14C18)`` is known to work. Other versions may also work.
- CMake. CMake 3.25.1 is known to work. Other versions may also work.
- (Optional) iPhone or iPad. This is for testing the app on your device.
  If you don't have a device, you can still run the app within a simulator on your Mac.

Download sherpa-ncnn
--------------------

First, let us download the source code of `sherpa-ncnn`_.

.. note::

  In the following, I will download `sherpa-ncnn`_ to
  ``$HOME/open-source``, i.e., ``/Users/fangjun/open-source``, on my Mac.

  You can put it anywhere as you like.

.. code-block:: bash

  mkdir -p $HOME/open-source
  cd $HOME/open-source
  git clone https://github.com/k2-fsa/sherpa-ncnn

Build sherpa-ncnn (in commandline, C++ Part)
--------------------------------------------

After downloading `sherpa-ncnn`_, let us build the C++ part of `sherpa-ncnn`_.

.. code-block:: bash

  cd $HOME/open-source/sherpa-ncnn/
  ./build-ios.sh

It will generate a directory
``$HOME/open-source/sherpa-ncnn/build-ios``, which we have already pre-configured
for you in Xcode.

.. hint::

  You don't have to look at the generated files in ``$HOME/open-source/sherpa-ncnn/build-ios``
  to build an app. We have pre-configured it for you.

  If you are eager to learn more about the generated files or want to use
  `sherpa-ncnn`_ in your own iOS project, please have a look
  at :ref:`sherpa_ncnn_ios_swift_for_the_more_curious`.


Build sherpa-ncnn (in Xcode)
----------------------------

Use the following command to open `sherpa-ncnn`_ in Xcode:

.. code-block:: bash

  cd $HOME/open-source/sherpa-ncnn/ios-swift/SherpaNcnn
  open SherpaNcnn.xcodeproj

It will start Xcode and you will see the following screenshot:

  .. figure:: ./pic/start-xcode-for-sherpa-ncnn.png
     :alt: Screenshot after running the command ``open SherpaNcnn.xcodeproj``
     :width: 600
     :align: center

     Screenshot after running the command ``open SherpaNcnn.xcodeproj``

Please select ``Product -> Build`` to build the project. See the screenshot
below:

  .. figure:: ./pic/select-product-build.png
     :alt: Screenshot for selecting ``Product -> Build``
     :width: 600
     :align: center

     Screenshot for selecting ``Product -> Build``

After finishing the build, you should see the following screenshot:

  .. figure:: ./pic/after-finishing-build.png
     :alt: Screenshot after finishing the build.
     :width: 100
     :align: center

     Screenshot after finishing the build.

Congratulations! You have successfully built the project. Let us run the
project by selecting ``Product -> Run``, which is shown in the following
screenshot:

  .. figure:: ./pic/run-the-project.png
     :alt: Screenshot for ``Product -> Run``.
     :width: 600
     :align: center

     Screenshot for ``Product -> Run``.

Please wait for a few seconds before Xcode starts the simulator.

Unfortunately, it will throw the following error:

  .. figure:: ./pic/error-no-model.png
     :alt: Screenshot for the error
     :width: 600
     :align: center

     Screenshot for the error

The reason for the above error is that we have not provided the pre-trained
model yet.

The file `ViewController.swift <https://github.com/k2-fsa/sherpa-ncnn/blob/master/ios-swift/SherpaNcnn/SherpaNcnn/ViewController.swift>`_
pre-selects the pre-trained model to be :ref:`sherpa-ncnn-mixed-english-chinese-conv-emformer-model`,
shown in the screenshot below:

  .. figure:: ./pic/pre-trained-model-1.png
     :alt: Screenshot for the pre-selected pre-trained model
     :width: 600
     :align: center

     Screenshot for the pre-selected pre-trained model

Let us add the pre-trained model :ref:`sherpa-ncnn-mixed-english-chinese-conv-emformer-model`
to Xcode. Please follow :ref:`sherpa-ncnn-mixed-english-chinese-conv-emformer-model`
to download it from `huggingface <https://huggingface.co/csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-06>`_.
You can download it to any directory as you like.

Please right click the project ``SherpaNcnn`` and select ``Add Files to "SherpaNcnn"...``
in the popup menu, as is shown in the screenshot below:

  .. figure:: ./pic/step-to-add-pre-trained-model-1.png
     :alt: Screenshot for adding files to SherpaNcnn
     :width: 600
     :align: center

     Screenshot for adding files to SherpaNcnn

In the popup dialog, switch to the folder where you just downloaded the pre-trained
model.

In the screenshot below, it is the
folder ``/Users/fangjun/open-source/icefall-models/sherpa-ncnn-conv-emformer-transducer-2022-12-06``:

  .. figure:: ./pic/step-to-add-pre-trained-model-2.png
     :alt: Screenshot for navigating to the folder containing the downloaded pre-trained
     :width: 600
     :align: center

     Screenshot for navigating to the folder containing the downloaded pre-trained

Select required files and click the button ``Add``:

  .. figure:: ./pic/step-to-add-pre-trained-model-3.png
     :alt: Screenshot for selecting required files
     :width: 600
     :align: center

     Screenshot for selecting required files

After adding pre-trained model files to Xcode, you should see the following
screenshot:

  .. figure:: ./pic/step-to-add-pre-trained-model-4.png
     :alt: Screenshot after add pre-trained model files
     :width: 600
     :align: center

     Screenshot after add pre-trained model files

At this point, you should be able to select the menu ``Product -> Run``
to run the project and you should finally see the following screenshot:

  .. figure:: ./pic/run.png
     :alt: Screenshot for a successful run.
     :width: 600
     :align: center

     Screenshot for a successful run.

Click the button to start recording! A screenshot is given below:

  .. figure:: ./pic/run-2.png
     :alt: Screenshot for recording and recognition.
     :width: 600
     :align: center

     Screenshot for recording and recognition.

Congratulations! You have finally succeeded in running `sherpa-ncnn`_ with iOS,
though it is in a simulator.

Please read below if you want to run `sherpa-ncnn`_ on your iPhone or iPad.

Run sherpa-ncnn on your iPhone/iPad
-----------------------------------

First, please make sure the iOS version of your iPhone/iPad is ``>= 13.0``.

Click the menu ``Xcode -> Settings...``, as is shown in the following screenshot:

  .. figure:: ./pic/xcode-settings.png
     :alt: Screenshot for ``Xcode -> Settings...``
     :width: 600
     :align: center

     Screenshot for ``Xcode -> Settings...``

In the popup dialog, please select ``Account`` and click ``+`` to add
your Apple ID, as is shown in the following ``screenshots``.

  .. figure:: ./pic/add-an-account.png
     :alt: Screenshot for selecting ``Account`` and click ``+``.
     :width: 600
     :align: center

     Screenshot for selecting ``Account`` and click ``+``.

  .. figure:: ./pic/add-an-account-2.png
     :alt: Screenshot for selecting ``Apple ID`` and click ``Continue``
     :width: 600
     :align: center

     Screenshot for selecting ``Apple ID`` and click ``Continue``

  .. figure:: ./pic/add-an-account-3.png
     :alt: Screenshot for adding your Apple ID and click ``Next``
     :width: 600
     :align: center

     Screenshot for adding your Apple ID and click ``Next``

  .. figure:: ./pic/add-an-account-4.png
     :alt: Screenshot for entering your password and click ``Next``
     :width: 600
     :align: center

     Screenshot for entering your password and click ``Next``

  .. figure:: ./pic/add-an-account-5.png
     :alt: Screenshot after adding your Apple ID
     :width: 600
     :align: center

     Screenshot after adding your Apple ID

After adding your Apple ID, please connect your iPhone or iPad to your Mac
and select your device in Xcode. The following screenshot is an example
to select my iPhone.

  .. figure:: ./pic/select-device.png
     :alt: Screenshot for selecting your device
     :width: 600
     :align: center

     Screenshot for selecting your device

Now your Xcode should look like below after selecting a device:

  .. figure:: ./pic/select-device-2.png
     :alt: Screenshot after selecting your device
     :width: 600
     :align: center

     Screenshot after selecting your device

Please select ``Product -> Run`` again to run `sherpa-ncnn`_ on your selected
device, as is shown in the following screenshot:

  .. figure:: ./pic/run-3.png
     :alt: Screenshot for selecting ``Product -> Run``
     :width: 600
     :align: center

     Screenshot for selecting ``Product -> Run``

After a successful build, check your iPhone/iPad and you should see the following
screenshot:

  .. figure:: ./pic/run-4.jpg
     :alt: Screenshot for running sherpa-ncnn on your device
     :width: 300
     :align: center

     Screenshot for running sherpa-ncnn on your device

To fix that, please select ``Settings -> General -> Device Management``
on your device

  .. figure:: ./pic/run-5.jpg
     :alt: Screenshot for selecting `Settings -> General -> Device Management` on your device
     :width: 300
     :align: center

     Screenshot for selecting `Settings -> General -> Device Management` on your device

Please click ``Apple Development: csukuangfj...`` and click ``Trust "Apple Development: csukuangfj@g..."``
in the subsequent dialog, as is shown below:

  .. figure:: ./pic/run-6.jpg
     :alt: Screenshot for "Trust "Apple Development: csukuangfj@g...""
     :width: 300
     :align: center

     Screenshot for "Trust "Apple Development: csukuangfj@g...""

At this point, you should be able to run the app on your device. The following is a screenshot
about running it on my iPhone:

  .. figure:: ./pic/run-7.jpg
     :alt: Screenshot for running `sherpa-ncnn`_ on iPhone
     :width: 300
     :align: center

     Screenshot for running `sherpa-ncnn`_ on iPhone


Congratulations! You have successfully run `sherpa-ncnn`_ on your device!
