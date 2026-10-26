# Simulation results (measured-distance error model)

## Error model

| Model | Pair | Same / different pairs | d' | Predicted Rank-1 (147 wrong candidates) | Correct with 1,000 candidates | False matches at one threshold (%) |
|---|---|---|---|---|---|---|
| DNDM | DD | 500 / 30625 | 3.12 | 84.5% | 75% | 5.4 |
| DNDM | ND | 1250 / 61250 | 3.04 | 80.3% | 63% | 3.9 |
| DNDM | NN | 500 / 30625 | 3.09 | 82.1% | 62% | 11.9 |
| SFM | DD | 500 / 30625 | 3.27 | 84.6% | 73% | 5.2 |
| SFM | ND | 1250 / 61250 | 3.18 | 82.9% | 69% | 4.8 |
| SFM | NN | 500 / 30625 | 3.46 | 85.6% | 70% | 4.7 |
| LPSMNet | DD | 500 / 30625 | 3.05 | 85.9% | 79% | 5.9 |
| LPSMNet | ND | 1250 / 61250 | 3.14 | 84.7% | 74% | 4.9 |
| LPSMNet | NN | 500 / 30625 | 3.26 | 87.4% | 71% | 5.6 |

## Evening period

Perfect sensing: TTS = 4088 veh·h. Guidance saves 11.5% of TTS (no guidance: +13.0% above perfect sensing; peak departures +5.1%, night departures +31.9%).
Effect of guidance on travel time by OD (perfect sensing vs. no guidance): OD1 -16.2%, OD2 +31.1%, OD3 +59.7%.

| Matching | Model | TTS above perfect (%) | Peak | Night | Kept (%) | Wrong peak/night (%) | Rejected peak/night (%) | Candidates peak/night |
|---|---|---|---|---|---|---|---|---|
| naive | DNDM | +7.3 ± 0.5 | +2.6 | +18.6 | 44 | 32/31 | 0/0 | 1240/633 |
| naive | SFM | +7.2 ± 0.4 | +3.7 | +15.7 | 45 | 29/25 | 0/0 | 1231/635 |
| naive | LPSMNet | +6.0 ± 0.4 | +2.7 | +14.0 | 54 | 26/23 | 0/0 | 1234/634 |
| traffic-aware | DNDM | +0.2 ± 0.0 | +0.3 | -0.0 | 98 | 7/3 | 22/30 | 230/64 |
| traffic-aware | SFM | +0.2 ± 0.0 | +0.2 | -0.0 | 99 | 6/2 | 21/24 | 229/64 |
| traffic-aware | LPSMNet | +0.1 ± 0.0 | +0.2 | +0.0 | 99 | 5/2 | 15/18 | 229/64 |

Paired TTS reduction of LPSMNet (% of the baseline TTS; positive = LPSMNet better):

| Matching | vs. | Whole period | p | Peak | Night |
|---|---|---|---|---|---|
| naive | DNDM | +1.19 | < 0.001 | -0.12 | +3.80 |
| naive | SFM | +1.10 | < 0.001 | +0.92 | +1.44 |
| traffic-aware | DNDM | +0.09 | < 0.001 | +0.14 | -0.02 |
| traffic-aware | SFM | +0.03 | 0.088 | +0.05 | -0.01 |

## Sensitivity study (evening, naive matching)

| Setting | No guidance | DNDM | SFM | LPSMNet | LPSMNet vs. SFM | LPSMNet vs. DNDM |
|---|---|---|---|---|---|---|
| Night demand ×0.6 | +4.3 | +3.3 | +3.0 | +2.4 | +0.49 (p 0.324) | +0.82 (p 0.093) |
| Base case | +12.4 | +6.8 | +6.8 | +5.5 | +1.29 (p 0.006) | +1.27 (p 0.008) |
| Night demand ×1.4 | +40.4 | +28.1 | +20.1 | +17.7 | +1.87 (p 0.116) | +8.09 (p < 0.001) |
| Compliance 0.4 | +13.2 | +8.0 | +8.0 | +6.6 | +1.28 (p 0.003) | +1.30 (p < 0.001) |
| Compliance 1.0 | -7.1 | -6.3 | -4.4 | -6.9 | +2.07 (p 0.167) | +0.23 (p 0.789) |
| θ = 1/240 s^-1 | +9.2 | +7.7 | +7.8 | +6.1 | +1.58 (p < 0.001) | +1.52 (p < 0.001) |
| θ = 1/60 s^-1 | +17.2 | +9.8 | +8.8 | +8.0 | +0.71 (p 0.145) | +1.60 (p 0.015) |
| Interval 30 s | +13.4 | +7.7 | +8.2 | +6.3 | +1.68 (p 0.001) | +1.27 (p < 0.001) |
| Interval 180 s | +10.8 | +6.8 | +6.4 | +5.3 | +1.02 (p 0.051) | +1.39 (p 0.005) |

## Morning period

Perfect sensing: TTS = 4658 veh·h. Guidance saves 10.6% of TTS (no guidance: +11.8% above perfect sensing; night departures +30.7%, peak departures +5.2%).

| Matching | Model | TTS above perfect (%) | Night | Peak | Kept (%) | Wrong night/peak (%) | Rejected night/peak (%) | Candidates night/peak |
|---|---|---|---|---|---|---|---|---|
| naive | DNDM | +8.2 ± 0.6 | +17.4 | +5.0 | 30 | 30/30 | 0/0 | 595/1205 |
| naive | SFM | +6.8 ± 0.6 | +14.3 | +4.2 | 42 | 24/28 | 0/0 | 596/1210 |
| naive | LPSMNet | +6.7 ± 0.5 | +13.8 | +4.2 | 43 | 22/24 | 0/0 | 595/1204 |
| traffic-aware | DNDM | +0.2 ± 0.1 | -0.0 | +0.3 | 98 | 3/6 | 30/21 | 64/263 |
| traffic-aware | SFM | +0.2 ± 0.0 | -0.0 | +0.3 | 98 | 2/6 | 24/20 | 64/263 |
| traffic-aware | LPSMNet | +0.1 ± 0.0 | -0.0 | +0.2 | 99 | 2/5 | 18/15 | 64/263 |

Paired TTS reduction of LPSMNet (% of the baseline TTS; positive = LPSMNet better):

| Matching | vs. | Whole period | p | Night | Peak |
|---|---|---|---|---|---|
| naive | DNDM | +1.40 | < 0.001 | +3.09 | +0.73 |
| naive | SFM | +0.10 | 0.599 | +0.41 | -0.04 |
| traffic-aware | DNDM | +0.07 | 0.005 | -0.01 | +0.10 |
| traffic-aware | SFM | +0.05 | 0.008 | -0.00 | +0.07 |

## TTS reduction of LPSMNet by departure hour (naive matching)

| Period | Hour | vs. SFM (%) | p | vs. DNDM (%) | p | Wrong matches LPSMNet / SFM / DNDM (%) |
|---|---|---|---|---|---|---|
| evening | 18:00 | -0.8 | 0.003 | -0.5 | 0.036 | 20.0 / 25.9 / 23.8 |
| evening | 19:00 | +0.5 | 0.184 | -2.7 | < 0.001 | 26.1 / 29.6 / 33.2 |
| evening | 20:00 | +3.3 | < 0.001 | +6.4 | < 0.001 | 31.4 / 31.4 / 40.1 |
| evening | 21:00 | +0.0 | 0.804 | +0.4 | 0.008 | 23.9 / 25.5 / 32.0 |
| evening | 22:00 | -0.3 | 0.083 | +0.8 | < 0.001 | 23.2 / 25.0 / 31.4 |
| evening | 23:00 | +1.1 | 0.048 | +2.5 | 0.001 | 22.4 / 24.2 / 30.4 |
| evening | 00:00 | +3.2 | < 0.001 | +7.6 | < 0.001 | 22.5 / 24.3 / 30.5 |
| evening | 01:00 | +0.1 | 0.521 | +1.6 | < 0.001 | 24.2 / 25.8 / 32.5 |
| morning | 00:00 | +0.1 | 0.412 | +0.5 | < 0.001 | 19.6 / 21.4 / 26.7 |
| morning | 01:00 | -0.3 | 0.712 | +0.9 | 0.146 | 22.3 / 24.1 / 30.4 |
| morning | 02:00 | +1.2 | 0.063 | +6.9 | < 0.001 | 22.4 / 24.1 / 30.3 |
| morning | 03:00 | +0.2 | 0.378 | +1.2 | < 0.001 | 24.2 / 25.9 / 32.8 |
| morning | 04:00 | +0.0 | 0.765 | +0.9 | < 0.001 | 23.3 / 24.9 / 31.3 |
| morning | 05:00 | -0.1 | 0.356 | +0.2 | 0.124 | 25.6 / 26.8 / 34.0 |
| morning | 06:00 | -1.3 | 0.003 | -0.1 | 0.782 | 25.4 / 28.8 / 32.1 |
| morning | 07:00 | +1.0 | 0.012 | +1.5 | < 0.001 | 21.7 / 29.0 / 26.4 |
