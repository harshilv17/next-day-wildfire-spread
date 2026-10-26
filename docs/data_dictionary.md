# Data Dictionary

Next Day Wildfire Spread dataset (Huot et al. 2022). Each sample is a 64 x 64 km patch at 1 km
resolution: 12 input channels on day t and the fire mask on day t+1 as the label.
Channel order is `CHANNELS` in `src/data.py`; `load_split` returns X with shape (N, 12, 64, 64).
Sources, units and native resolutions are from `notebooks/notes_dataset.md`.

| # | Name | Description | Unit | Source dataset | Native resolution | Value range (p1 to p99)¹ |
| -: | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | `elevation` | Terrain height | m | SRTM | 30 m | 2 to 2,954 |
| 1 | `th` | Wind direction, clockwise from north, direction the wind blows from (daily mean) | degrees | GRIDMET | 4 km | 31 to 331² |
| 2 | `vs` | Wind speed (daily mean) | m/s | GRIDMET | 4 km | 1.39 to 7.54 |
| 3 | `tmmn` | Minimum temperature | K | GRIDMET | 4 km | 264.7 to 297.1 |
| 4 | `tmmx` | Maximum temperature | K | GRIDMET | 4 km | 277.6 to 312.7 |
| 5 | `sph` | Specific humidity | kg/kg | GRIDMET | 4 km | 0.0012 to 0.0178 |
| 6 | `pr` | Precipitation (daily total) | mm | GRIDMET | 4 km | -0.02 to 7.52³ |
| 7 | `pdsi` | Palmer drought severity index (5-day composite) | index | GRIDMET Drought | 4 km | -5.14 to 5.25 |
| 8 | `NDVI` | Normalised difference vegetation index (8-day composite), stored x10^4 | index x10^4 | VIIRS VNP13A1 | 500 m | 0 to 8,936⁴ |
| 9 | `population` | Population density | people/km² | GPWv4 | 1 km | 0 to 851 |
| 10 | `erc` | Energy release component (fire danger index) | index | NFDRS via GRIDMET | 4 km | 11.3 to 104.1 |
| 11 | `PrevFireMask` | Active fire mask on day t | class | MODIS MOD14A1 V6 | 1 km | {-1, 0, 1}⁵ |
| label | `FireMask` | Active fire mask on day t+1 (prediction target) | class | MODIS MOD14A1 V6 | 1 km | {-1, 0, 1}⁵ |

## Notes

1. 1st to 99th percentile over all pixels of the training split (14,979 patches, 61.4 M pixels),
   computed with `np.percentile` on the output of `load_split('train')`. We report percentiles,
   not min/max, because the raw export contains artefacts such as wind direction down to
   -505,870 degrees and temperatures up to 1,230 K. Observed min/max, the share of out-of-range
   pixels and the fix for each channel are in the table in
   [`reports/data_quality.md`](../reports/data_quality.md).
2. 0.23% of `th` values lie outside [0, 360). `src/features.py` wraps them with `th % 360`
   instead of clipping, because direction is circular. After wrapping, p1 to p99 is 32 to 329.
3. 9.8% of `pr` values are negative (an export artefact); `src/features.py` clips `pr` at 0.
4. Divide by 10,000 for the usual NDVI scale (p1 to p99 = 0 to 0.89). Negative values (0.9% of
   pixels) are real (water, snow) and are not clipped.
5. Fire masks: `1` = fire, `0` = no fire, `-1` = uncertain (cloud, smoke or missing data).
   Pixels with `-1` in `FireMask` are ignored in training and in all metrics. On the training
   split 1.1% of labelled `FireMask` pixels are fire (90:1) and 2.3% of pixels are `-1`.
