## Introduction

An ASR server framework supporting both streaming and non-streaming recognition.

Most parts will be implemented in Python, while CPU-bound tasks are implemented
in C++, which are called by Python threads with the GIL being released.

## TODOs

- [ ] Support non-streaming recognition
- [ ] Documentation for installation and usage
- [ ] Support streaming recognition
