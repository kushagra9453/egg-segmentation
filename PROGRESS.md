# PROGRESS

- [x] STEP 1: scripts/15_track.py created – tracking generated (3409 frames, 69 unique IDs, outputs/tracking_output.mp4)
- [x] STEP 2: T09 and T10 trials added to scripts/14_classical.py – T10 selected as final method (Dev IoU 0.667)
- [x] STEP 3: Task 3 evidence generated – boundary F-scores in CSVs, scene_iou.png, and outputs/results/analysis.md
- [x] STEP 4: README.md, report/report.md, requirements.txt refreshed and committed
- [x] STEP 5: T11 debugged – verified K=1000 sanity check matches T10 (diff 0.0001 < 0.01); sweep {1.5..1000} logged to experiment_log.csv
- [x] STEP 6: T12 added (distance-transform opening) – F sweep {0.3..0.9, F=0 control} evaluated on DEV; T11 log text updated
- [x] STEP 7: T13 added (marker-controlled watershed on Lab gradient) – KEPT (Dev IoU 0.681); T14 logged
- [x] STEP 8: T15 added (egg-radius scaled GrabCut bands) – KEPT (Dev IoU 0.707); T16-T18 logged; results & docs updated
- [x] STEP 9: T19 applied (Felzenszwalb candidates) – KEPT (Dev IoU 0.729, Eval 0.711); T20 logged; results & docs updated
- [x] STEP 10: T21 applied (tightened multi-egg union 0.85) – KEPT (Dev IoU 0.766, Eval 0.764); results & docs updated
