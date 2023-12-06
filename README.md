# ETL Design repository

This repository is a collection of tools to optimize the ETL layout and run simple acceptance studies.

## Usage

`ETL.py` has the main code and classes for the ETL geometry.

Studies to populate Dees can be run with 
``` shell
ipython -i dee_geometry.py -- --modules L --dee_layout updated
```
Options are:
``` shell
optional arguments:
  -h, --help            show this help message and exit
  --skip_acceptance     Don't run the acceptance studies
  --comparison          Make a comparison plot of the different scenarios
  --modules {S,M,L}     Module size
  --dee_layout {baseline,updated,plain,updatedV2}
                        Select which dee layout to use
  --seal                Add space for seal
  --no_feedthrough      Don't put the geometry for feedthrough
```

## Assumptions

Several assumption about LGAD sensors are made:
- Formulas for the radiation dependence are all in [sensors.py](sensors.py).
- Every sensor is assumed to draw 0.75mA from the BV supply coming from surface effects etc (independent of irradiation), which dominates over the per-pixel leakage current of up to 0.25mA at end-of-life. 
These numbers were provided by Nicolo, and account for safety margins.
