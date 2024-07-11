# ECG Pathology Classifier
> Utilizes a CNN model to classify heart conditions in 5 categories using a 12-lead ECG

A model created to be the base of my final paper at college. It utilizes PTB-XL database (Wagner et al., 2022, available at: https://physionet.org/content/ptb-xl/1.0.3/) as training and testing of the model.
The database is made of almost 22.000 ECG exams, each one being represented by 10 second recordings of each one of the 12 leads of an ECG (I, II, III, AVL, AVR, AVF, V1, V2, V3, V4, V5, V6 respectively).
The CNN model used is based on examples of timeseries classifications from keras.

## Development setup

To use the algorithm:
- Download the repsitory available at: https://physionet.org/content/ptb-xl/1.0.3/;
- Unzip the PTB-XL archive to the same folder:
- Install the following dependencies/libraries if not installed:

```sh
pandas
wfdb
numpy
ast
matplotlib
tensorflow
scykit-learn
seaborn
imblearn
```

## Release History

* 0.0.1
    * Work in progress, soon a version utillizing Resnet-50 will be posted as well

## Meta

Thomas Sponchiado Pastore – [@ThomasSPastore](https://twitter.com/dbader_org) – thomas.spastore@gmail.com

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/thom01s?tab=repositories](https://github.com/dbader/)

## Contributing

1. Fork it (<https://github.com/thom01s/Heart-pathology-classification/fork>)
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

<!-- Markdown link & img dfn's -->
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki
